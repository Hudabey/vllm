# SPDX-License-Identifier: Apache-2.0
"""tollbooth: exact trace dump for the DSA lightning-indexer selection.

One record per (query row, indexer layer, TP rank): the selector's top-k
positions verbatim, their FP32 index scores gathered from the live logits,
tau = min over valid selected scores, and the exact prefix identity via an
append-only token ledger. The hook is a pure reader of the selector's output;
it never reselects, sorts, or writes into the selector's buffers.

Layer-path rules (``TraceSession.capture``):
  * device ops only: gather / where / amin / sum / isfinite / copy_ / event
  * no ``.cpu()``, ``.item()``, ``.tolist()``, no disk I/O, no host loops over rows
  * the only blocking point is ring backpressure on a *completed-copy event*

Everything here runs on CPU tensors under pytest; CUDA only changes which
event/stream/allocator is used.

Record layout (little endian, 64 + 8k bytes; 16,448 at k = 2048):
  header  <QQQQIIIHHHHfII>
      run_id, step_id, request_key, prefix_node_id            4 x u64
      query_position, query_token_id, prefix_length           3 x u32
      layer_id, tp_rank, valid_count, flags                   4 x u16
      tau                                                     f32
      attempt_id, reserved                                    2 x u32
  ids     int32[k]   selector output, original order, -1 padding preserved
  scores  float32[k] logits gathered at ids, NaN at -1
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import queue
import struct
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Protocol

import numpy as np
import torch

SCHEMA_VERSION = 1

HEADER = struct.Struct("<QQQQIIIHHHHfII")
assert HEADER.size == 64

HEADER_DTYPE = np.dtype(
    [
        ("run_id", "<u8"),
        ("step_id", "<u8"),
        ("request_key", "<u8"),
        ("prefix_node_id", "<u8"),
        ("query_position", "<u4"),
        ("query_token_id", "<u4"),
        ("prefix_length", "<u4"),
        ("layer_id", "<u2"),
        ("tp_rank", "<u2"),
        ("valid_count", "<u2"),
        ("flags", "<u2"),
        ("tau", "<f4"),
        ("attempt_id", "<u4"),
        ("reserved", "<u4"),
    ]
)
assert HEADER_DTYPE.itemsize == 64

LEDGER_EDGE = struct.Struct("<QI")  # parent_node, token
LEDGER_EDGE_DTYPE = np.dtype([("parent", "<u8"), ("token", "<u4")])


def record_dtype(k: int) -> np.dtype:
    return np.dtype(
        [("header", HEADER_DTYPE), ("ids", "<i4", (k,)), ("scores", "<f4", (k,))]
    )


def record_size(k: int) -> int:
    return 64 + 8 * k


class Flag(IntEnum):
    DECODE = 1  # phase: 0 prefill, 1 decode
    SHADOW = 2  # score_source: 0 selector, 1 shadow_dense
    TAU_VALID = 4
    VIOLATION = 8  # contract violation detected on device for this row
    # bits 4-5: SampleMode of the run that produced the record (see SAMPLE_SHIFT)


SAMPLE_SHIFT = 4
SAMPLE_MASK = 0x3 << SAMPLE_SHIFT


class SampleMode(IntEnum):
    FULL = 0  # every row of every capture
    DECODE_ONLY = 1  # decode rows only; prefill captures are skipped entirely
    SAMPLED = 2  # decode rows; prefill rows with query_position % every == 0
    #               plus the final `tail` rows of each chunk
    WINDOWED = 3  # only rows whose (request prompt hash, absolute query position) fall in
    #               a configured window; applies to prefill and decode rows alike

    @classmethod
    def parse(cls, name: str) -> "SampleMode":
        return cls[name.strip().upper()]


def sample_mode_of(flags: int) -> SampleMode:
    return SampleMode((int(flags) & SAMPLE_MASK) >> SAMPLE_SHIFT)


class Phase(IntEnum):
    PREFILL = 0
    DECODE = 1


class ScoreSource(IntEnum):
    SELECTOR = 0
    SHADOW_DENSE = 1


class TraceContractError(RuntimeError):
    """A trace record violated the exactness contract (non-finite valid score,
    out-of-prefix id, or valid_count != min(k, prefix_len))."""


# --------------------------------------------------------------------------- #
# Prefix ledger
# --------------------------------------------------------------------------- #


class PrefixLedger:
    """Exact parent/token interning; node 0 is the empty prefix.

    Hashes are lookup aids, not identity proofs. New edges are queued in
    ``pending`` so the writer can persist them before any record that
    references them.
    """

    def __init__(self) -> None:
        self.nodes: list[tuple[int | None, int | None]] = [(None, None)]
        self.edges: dict[tuple[int, int], int] = {}
        self.pending: list[tuple[int, int, int]] = []  # (node, parent, token)

    def append(self, parent: int, token: int) -> int:
        key = (parent, token)
        node = self.edges.get(key)
        if node is None:
            node = len(self.nodes)
            self.edges[key] = node
            self.nodes.append(key)
            self.pending.append((node, parent, token))
        return node

    def add(self, tokens: Sequence[int]) -> int:
        node = 0
        for token in tokens:
            node = self.append(node, int(token))
        return node

    def add_path(self, tokens: Sequence[int]) -> list[int]:
        """Node id for every prefix length 1..len(tokens), in order."""
        node = 0
        out = []
        for token in tokens:
            node = self.append(node, int(token))
            out.append(node)
        return out

    def add_path_from(self, node: int, tokens: Sequence[int]) -> list[int]:
        """Extend an existing prefix node by ``tokens``; node id per new length."""
        out = []
        for token in tokens:
            node = self.append(node, int(token))
            out.append(node)
        return out

    def tokens(self, node: int) -> list[int]:
        result: list[int] = []
        while node:
            node, token = self.nodes[node]  # type: ignore[assignment]
            result.append(token)  # type: ignore[arg-type]
        return result[::-1]

    def drain_pending(self) -> list[tuple[int, int, int]]:
        out, self.pending = self.pending, []
        return out

    @staticmethod
    def edges_to_bytes(edges: Sequence[tuple[int, int, int]]) -> bytes:
        arr = np.empty(len(edges), dtype=LEDGER_EDGE_DTYPE)
        for i, (_node, parent, token) in enumerate(edges):
            arr[i] = (parent, token)
        return arr.tobytes()

    @classmethod
    def from_file(cls, path: str) -> "PrefixLedger":
        """Rebuild a ledger from its edge file; node id == 1 + edge index."""
        ledger = cls()
        with open(path, "rb") as f:
            arr = np.frombuffer(f.read(), dtype=LEDGER_EDGE_DTYPE)
        for parent, token in arr:
            node = ledger.append(int(parent), int(token))
            assert node == len(ledger.nodes) - 1, "ledger file is not append-only"
        ledger.pending.clear()
        return ledger


# --------------------------------------------------------------------------- #
# Per-forward context (host side; built by the model runner)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class RowMeta:
    request_key: int
    prefix_node_id: int  # ledger node of tokens[0 : query_position + 1]
    query_position: int
    query_token_id: int
    attempt_id: int = 0
    phase: int = Phase.PREFILL
    prompt_hash: str = ""  # sha256 of the request's prompt token ids (int32 LE bytes)


def prompt_hash_of(token_ids) -> str:
    import numpy as _np
    arr = _np.asarray(token_ids, dtype="<i4")
    return hashlib.sha256(arr.tobytes()).hexdigest()


@dataclass
class TraceContext:
    """One per forward pass, per rank. ``rows[i]`` describes token row ``i`` of
    the forward's flattened token order (after any batch reordering)."""

    run_id: int
    step_id: int
    tp_rank: int
    tp_world_size: int
    rows: list[RowMeta]


# --------------------------------------------------------------------------- #
# Device-side gather (the only code on the layer path)
# --------------------------------------------------------------------------- #


@dataclass
class Gathered:
    ids: torch.Tensor  # int32 [R, k], owned clone of the selector's slice
    scores: torch.Tensor  # f32   [R, k], NaN at -1
    tau: torch.Tensor  # f32   [R], NaN when valid_count < k
    valid_count: torch.Tensor  # int32 [R]
    violations: torch.Tensor  # int32 [R]
    prefix_len: torch.Tensor  # int32 [R]


def gather_scores(
    logits: torch.Tensor,
    topk_indices: torch.Tensor,
    row_starts: torch.Tensor,
    prefix_lens: torch.Tensor,
    k: int,
) -> Gathered:
    """Gather the selector's scores without reselecting or sorting.

    logits:        f32 [R, C] live score matrix (read only)
    topk_indices:  int32 [R, k] selector output (read only), -1 padding
    row_starts:    int32 [R] column offset of each row's causal window
                   (chunk.cu_seqlen_ks for prefill, zeros for decode)
    prefix_lens:   int32 [R] number of keys the selector scored for the row
    """
    if logits.dim() != 2 or topk_indices.dim() != 2:
        raise ValueError("logits and topk_indices must be 2-D")
    if topk_indices.shape[1] != k:
        raise ValueError(f"topk_indices has {topk_indices.shape[1]} columns, k={k}")
    if logits.shape[0] != topk_indices.shape[0]:
        raise ValueError("logits and topk_indices row counts differ")

    ids = topk_indices.clone()  # owned snapshot; the buffer is reused next layer
    mask = ids >= 0
    row_starts = row_starts.to(dtype=torch.int64)
    prefix_lens = prefix_lens.to(dtype=torch.int32)
    ids64 = ids.to(torch.int64)
    in_range = mask & (ids64 < prefix_lens[:, None].to(torch.int64))
    # Clamp so the gather itself can never fault; out-of-range rows are flagged.
    col = torch.where(in_range, row_starts[:, None] + ids64, torch.zeros_like(ids64))
    ncol = logits.shape[1]
    col = col.clamp_(0, max(ncol - 1, 0))
    if ncol == 0:
        scores = torch.full(ids.shape, float("nan"), dtype=torch.float32, device=ids.device)
    else:
        scores = logits.gather(1, col)
        scores = torch.where(mask, scores, torch.full_like(scores, float("nan")))

    valid_count = mask.sum(dim=1, dtype=torch.int32)
    nonfinite = (in_range & ~torch.isfinite(scores)).sum(dim=1, dtype=torch.int32)
    out_of_range = (mask & ~in_range).sum(dim=1, dtype=torch.int32)
    expected = torch.minimum(prefix_lens, torch.full_like(prefix_lens, k))
    count_bad = (valid_count != expected).to(torch.int32)
    violations = nonfinite + out_of_range + count_bad

    tau_valid = valid_count == k
    inf = torch.full_like(scores, float("inf"))
    tau_all = torch.where(mask, scores, inf).amin(dim=1)
    tau = torch.where(tau_valid, tau_all, torch.full_like(tau_all, float("nan")))
    return Gathered(ids, scores, tau, valid_count, violations, prefix_lens)


# --------------------------------------------------------------------------- #
# Events / streams (CUDA or CPU)
# --------------------------------------------------------------------------- #


class EventLike(Protocol):
    def record(self, stream=None) -> None: ...
    def synchronize(self) -> None: ...
    def query(self) -> bool: ...


class _CpuEvent:
    def __init__(self) -> None:
        self._done = False

    def record(self, stream=None) -> None:
        self._done = True

    def synchronize(self) -> None:
        pass

    def query(self) -> bool:
        return self._done


def _make_event(device: torch.device) -> EventLike:
    if device.type == "cuda":
        return torch.cuda.Event(blocking=False)
    return _CpuEvent()


# --------------------------------------------------------------------------- #
# Pinned ring
# --------------------------------------------------------------------------- #


class SlotState(IntEnum):
    FREE = 0
    FILLING = 1
    QUEUED = 2
    WRITING = 3


@dataclass
class RingSlot:
    index: int
    capacity_rows: int
    h_ids: torch.Tensor  # int32 [cap, k] pinned
    h_scores: torch.Tensor  # f32   [cap, k] pinned
    h_tau: torch.Tensor  # f32   [cap]    pinned
    h_valid_count: torch.Tensor  # int32 [cap]    pinned
    h_violations: torch.Tensor  # int32 [cap]    pinned
    h_prefix_len: torch.Tensor  # int32 [cap]    pinned
    producer_event: EventLike
    copy_event: EventLike
    rows: int = 0
    meta: Sequence[RowMeta] = ()
    step_id: int = 0
    layer_id: int = 0
    phase: int = Phase.PREFILL
    score_source: int = ScoreSource.SELECTOR
    sample_mode: int = SampleMode.FULL
    ledger_edges: list[tuple[int, int, int]] = field(default_factory=list)
    device_refs: tuple = ()  # keeps Gathered tensors alive until copy_event completes
    state: SlotState = SlotState.FREE


class TraceRing:
    """Fixed pool of pinned host slots. The producer never drops a record: if
    every slot is busy it waits on the oldest queued slot's copy event and then
    on the writer releasing it (counted in ``stalls``)."""

    def __init__(
        self,
        num_slots: int,
        capacity_rows: int,
        k: int,
        device: torch.device,
        pin_memory: bool | None = None,
    ) -> None:
        if num_slots < 1 or capacity_rows < 1:
            raise ValueError("ring needs at least one slot and one row")
        self.k = k
        self.device = device
        self.capacity_rows = capacity_rows
        pin = (device.type == "cuda") if pin_memory is None else pin_memory
        self.slots: list[RingSlot] = []
        for i in range(num_slots):
            mk = lambda shape, dt: torch.empty(shape, dtype=dt, pin_memory=pin)  # noqa: E731
            self.slots.append(
                RingSlot(
                    index=i,
                    capacity_rows=capacity_rows,
                    h_ids=mk((capacity_rows, k), torch.int32),
                    h_scores=mk((capacity_rows, k), torch.float32),
                    h_tau=mk((capacity_rows,), torch.float32),
                    h_valid_count=mk((capacity_rows,), torch.int32),
                    h_violations=mk((capacity_rows,), torch.int32),
                    h_prefix_len=mk((capacity_rows,), torch.int32),
                    producer_event=_make_event(device),
                    copy_event=_make_event(device),
                )
            )
        self._free: queue.SimpleQueue[RingSlot] = queue.SimpleQueue()
        for s in self.slots:
            self._free.put(s)
        self.queued: queue.Queue[RingSlot | None] = queue.Queue()
        self._inflight: list[RingSlot] = []
        self._lock = threading.Lock()
        self.stalls = 0
        self.failed: BaseException | None = None  # set by fail(); acquire raises
        self.max_queued = 0
        self._occ_sum = 0
        self._occ_n = 0
        self.copy_stream = torch.cuda.Stream() if device.type == "cuda" else None

    def fail(self, exc: BaseException) -> None:
        """Mark the ring broken (writer died). Every slot still queued is
        released so a producer blocked on a free slot wakes up and sees the
        failure instead of waiting forever."""
        if self.failed is None:
            self.failed = exc
        while True:
            try:
                slot = self.queued.get_nowait()
            except queue.Empty:
                break
            if slot is not None:
                self.release(slot)

    def _raise_if_failed(self) -> None:
        if self.failed is not None:
            raise TraceContractError(
                "trace writer failed; ring is closed"
            ) from self.failed

    def occupancy(self) -> dict:
        return {
            "max_queued": self.max_queued,
            "mean_queued_at_submit": (self._occ_sum / self._occ_n) if self._occ_n else 0.0,
            "submits": self._occ_n,
            "stalls": self.stalls,
        }

    def acquire(self, rows: int) -> RingSlot:
        if rows > self.slots[0].capacity_rows:
            raise ValueError(f"{rows} rows exceed slot capacity {self.slots[0].capacity_rows}")
        self._raise_if_failed()
        try:
            slot = self._free.get_nowait()
        except queue.Empty:
            # Backpressure: wait for the oldest in-flight copy, then for the
            # writer to release a slot. Blocks on an event, never on .item().
            # The wait is bounded per iteration so a writer failure (which
            # releases every queued slot and sets self.failed) is observed.
            self.stalls += 1
            with self._lock:
                oldest = self._inflight[0] if self._inflight else None
            if oldest is not None:
                oldest.copy_event.synchronize()
            while True:
                self._raise_if_failed()
                try:
                    slot = self._free.get(timeout=0.05)
                    break
                except queue.Empty:
                    continue
        if self.failed is not None:
            self._free.put(slot)
            self._raise_if_failed()
        slot.state = SlotState.FILLING
        slot.rows = rows
        return slot

    def submit(self, slot: RingSlot, g: Gathered) -> None:
        """Enqueue async D2H copies of ``g`` into ``slot`` and hand it to the writer."""
        n = slot.rows
        if g.ids.shape[0] != n:
            raise ValueError("gathered row count does not match slot")
        pairs = (
            (slot.h_ids[:n], g.ids),
            (slot.h_scores[:n], g.scores),
            (slot.h_tau[:n], g.tau),
            (slot.h_valid_count[:n], g.valid_count),
            (slot.h_violations[:n], g.violations),
            (slot.h_prefix_len[:n], g.prefix_len),
        )
        slot.device_refs = (g.ids, g.scores, g.tau, g.valid_count, g.violations, g.prefix_len)
        if self.copy_stream is not None:
            cur = torch.cuda.current_stream(self.device)
            slot.producer_event.record(cur)
            with torch.cuda.stream(self.copy_stream):
                self.copy_stream.wait_event(slot.producer_event)
                for dst, src in pairs:
                    dst.copy_(src, non_blocking=True)
                slot.copy_event.record(self.copy_stream)
        else:
            slot.producer_event.record()
            for dst, src in pairs:
                dst.copy_(src)
            slot.copy_event.record()
        slot.state = SlotState.QUEUED
        with self._lock:
            self._inflight.append(slot)
        q = self.queued.qsize()
        self.max_queued = max(self.max_queued, q + 1)
        self._occ_sum += q
        self._occ_n += 1
        self.queued.put(slot)

    def release(self, slot: RingSlot) -> None:
        with self._lock:
            if slot in self._inflight:
                self._inflight.remove(slot)
        slot.device_refs = ()
        slot.meta = ()
        slot.ledger_edges = []
        slot.rows = 0
        slot.state = SlotState.FREE
        self._free.put(slot)


# --------------------------------------------------------------------------- #
# Serialization
# --------------------------------------------------------------------------- #


def serialize_slot(slot: RingSlot, run_id: int, tp_rank: int, k: int) -> tuple[bytes, int]:
    """Pack a completed slot into record bytes. Copies out of pinned memory
    (``tobytes``) so the slot can be reused immediately afterwards.
    Returns (bytes, number of rows with violations)."""
    n = slot.rows
    rec = np.zeros(n, dtype=record_dtype(k))
    hdr = rec["header"]
    meta = slot.meta
    if len(meta) != n:
        raise ValueError(f"slot has {n} rows but {len(meta)} row metas")
    hdr["run_id"] = run_id
    hdr["step_id"] = slot.step_id
    hdr["request_key"] = [m.request_key for m in meta]
    hdr["prefix_node_id"] = [m.prefix_node_id for m in meta]
    hdr["query_position"] = [m.query_position for m in meta]
    hdr["query_token_id"] = [m.query_token_id for m in meta]
    hdr["attempt_id"] = [m.attempt_id for m in meta]
    hdr["layer_id"] = slot.layer_id
    hdr["tp_rank"] = tp_rank
    hdr["reserved"] = SCHEMA_VERSION

    valid = slot.h_valid_count[:n].numpy()
    viol = slot.h_violations[:n].numpy()
    hdr["valid_count"] = np.minimum(valid, np.iinfo(np.uint16).max).astype(np.uint16)
    hdr["prefix_length"] = slot.h_prefix_len[:n].numpy().astype(np.uint32)
    hdr["tau"] = slot.h_tau[:n].numpy()
    flags = np.zeros(n, dtype=np.uint16)
    if slot.phase == Phase.DECODE:
        flags |= np.uint16(Flag.DECODE)
    if slot.score_source == ScoreSource.SHADOW_DENSE:
        flags |= np.uint16(Flag.SHADOW)
    flags |= np.where(valid == k, np.uint16(Flag.TAU_VALID), np.uint16(0))
    flags |= np.where(viol > 0, np.uint16(Flag.VIOLATION), np.uint16(0))
    flags |= np.uint16(int(slot.sample_mode) << SAMPLE_SHIFT)
    hdr["flags"] = flags
    rec["ids"] = slot.h_ids[:n].numpy()
    rec["scores"] = slot.h_scores[:n].numpy()
    return rec.tobytes(), int((viol > 0).sum())


def read_records(path: str, k: int) -> np.ndarray:
    with open(path, "rb") as f:
        return np.frombuffer(f.read(), dtype=record_dtype(k)).copy()


# --------------------------------------------------------------------------- #
# Writer thread + session
# --------------------------------------------------------------------------- #


class TraceWriter(threading.Thread):
    def __init__(self, session: "TraceSession") -> None:
        super().__init__(name=f"tollbooth-writer-r{session.tp_rank}", daemon=True)
        self.s = session
        self.records_written = 0
        self.bytes_written = 0

    def run(self) -> None:
        ring = self.s.ring
        while True:
            slot = ring.queued.get()
            if slot is None:
                break
            try:
                slot.state = SlotState.WRITING
                slot.copy_event.synchronize()
                if slot.ledger_edges:
                    self.s.ledger_fd.write(PrefixLedger.edges_to_bytes(slot.ledger_edges))
                    self.s.ledger_fd.flush()
                data, n_viol = serialize_slot(slot, self.s.run_id, self.s.tp_rank, self.s.k)
                self.s.records_fd.write(data)
                self.s.records_fd.flush()
                self.records_written += slot.rows
                self.bytes_written += len(data)
                if n_viol:
                    # The offending records are on disk with the VIOLATION flag
                    # for postmortem; now fail loudly.
                    raise TraceContractError(
                        f"rank {self.s.tp_rank} step {slot.step_id} layer {slot.layer_id}: "
                        f"{n_viol} row(s) with non-finite valid scores, out-of-prefix ids, "
                        f"or valid_count != min(k, prefix_len)"
                    )
            except BaseException as e:  # noqa: BLE001
                self.s.fatal = e
                ring.release(slot)
                ring.fail(e)  # unblock any producer waiting on a slot
                break
            ring.release(slot)


class TraceSession:
    """Owns the ring, the writer thread, the ledger and the output files for
    one TP rank. ``capture`` is the layer-path entry point."""

    def __init__(
        self,
        out_dir: str,
        run_id: int,
        tp_rank: int,
        tp_world_size: int,
        k: int,
        device: torch.device | str = "cpu",
        ring_slots: int = 64,
        capacity_rows: int = 8192,
        pin_memory: bool | None = None,
        manifest_extra: dict | None = None,
        sample_mode: SampleMode | int = SampleMode.FULL,
        sample_every: int = 64,
        sample_tail: int = 256,
        windows: dict[str, list[tuple[int, int]]] | None = None,
    ) -> None:
        self.out_dir = out_dir
        # WINDOWED: prompt_hash -> list of inclusive [start, end] absolute query positions
        self.windows: dict[str, list[tuple[int, int]]] = {
            k: [(int(a), int(b)) for a, b in v] for k, v in (windows or {}).items()
        }
        self.window_rows_seen = 0
        self.window_rows_captured = 0
        self.sample_mode = SampleMode(int(sample_mode))
        self.sample_every = int(sample_every)
        self.sample_tail = int(sample_tail)
        self.prefill_rows_seen = 0
        self.prefill_rows_captured = 0
        self.run_id = run_id
        self.tp_rank = tp_rank
        self.tp_world_size = tp_world_size
        self.k = k
        self.device = torch.device(device)
        self.ring = TraceRing(ring_slots, capacity_rows, k, self.device, pin_memory)
        self.ledger = PrefixLedger()
        self.request_keys: dict[int, str] = {}  # request_key -> engine request id
        self.fatal: BaseException | None = None
        self.manifest_extra = manifest_extra or {}
        self.captures = 0
        os.makedirs(out_dir, exist_ok=True)
        self.records_path = os.path.join(out_dir, f"tollbooth.rank{tp_rank}.records")
        self.ledger_path = os.path.join(out_dir, f"tollbooth.rank{tp_rank}.ledger")
        self.manifest_path = os.path.join(out_dir, f"tollbooth.rank{tp_rank}.manifest.json")
        self.records_fd = open(self.records_path, "ab")
        self.ledger_fd = open(self.ledger_path, "ab")
        self._started = time.time()
        self.writer = TraceWriter(self)
        self.writer.start()

    # ---- layer path ----------------------------------------------------- #

    def capture(
        self,
        ctx: TraceContext | None,
        layer_id: int,
        logits: torch.Tensor,
        topk_indices: torch.Tensor,
        row_starts: torch.Tensor,
        prefix_lens: torch.Tensor,
        rows: slice | Sequence[int],
        phase: int = Phase.PREFILL,
        score_source: int = ScoreSource.SELECTOR,
    ) -> None:
        """Record the selector output for ``logits``/``topk_indices`` rows.

        ``rows`` maps logits row i -> ctx.rows index (a slice for contiguous
        chunks, an explicit index list for padded decode rows). No-op when
        ``ctx`` is None (dummy / profiling / graph-capture runs).
        """
        if ctx is None:
            return
        if self.fatal is not None:
            raise TraceContractError("trace writer failed earlier") from self.fatal
        meta = ctx.rows[rows] if isinstance(rows, slice) else [ctx.rows[i] for i in rows]
        n = logits.shape[0]
        if len(meta) != n:
            raise ValueError(f"{n} logits rows but {len(meta)} row metas")
        if n == 0:
            return
        if self.sample_mode == SampleMode.WINDOWED:
            self.window_rows_seen += n
            keep = self.windowed_rows(meta)
            if not keep:
                return
            if len(keep) < n:
                idx = torch.tensor(keep, dtype=torch.int64, device=logits.device)
                logits = logits.index_select(0, idx)
                topk_indices = topk_indices.index_select(0, idx)
                row_starts = row_starts.index_select(0, idx)
                prefix_lens = prefix_lens.index_select(0, idx)
                meta = [meta[i] for i in keep]
                n = len(keep)
            self.window_rows_captured += n
        elif phase == Phase.PREFILL and self.sample_mode != SampleMode.FULL:
            self.prefill_rows_seen += n
            if self.sample_mode == SampleMode.DECODE_ONLY:
                return
            keep = self.sampled_prefill_rows(meta, n)
            if len(keep) == 0:
                return
            if len(keep) < n:
                idx = torch.tensor(keep, dtype=torch.int64, device=logits.device)
                logits = logits.index_select(0, idx)
                topk_indices = topk_indices.index_select(0, idx)
                row_starts = row_starts.index_select(0, idx)
                prefix_lens = prefix_lens.index_select(0, idx)
                meta = [meta[i] for i in keep]
                n = len(keep)
            self.prefill_rows_captured += n
        g = gather_scores(logits, topk_indices, row_starts, prefix_lens, self.k)
        cap = self.ring.capacity_rows
        edges = self.ledger.drain_pending()
        for start in range(0, n, cap):  # split batches larger than one slot
            end = min(start + cap, n)
            slot = self.ring.acquire(end - start)
            slot.meta = meta[start:end]
            slot.step_id = ctx.step_id
            slot.layer_id = layer_id
            slot.phase = phase
            slot.score_source = score_source
            slot.sample_mode = self.sample_mode
            slot.ledger_edges = edges
            edges = []
            part = g if (start == 0 and end == n) else Gathered(
                g.ids[start:end], g.scores[start:end], g.tau[start:end],
                g.valid_count[start:end], g.violations[start:end],
                g.prefix_len[start:end])
            self.ring.submit(slot, part)
        self.captures += 1

    def windowed_rows(self, meta: Sequence[RowMeta]) -> list[int]:
        """Row indices whose request prompt hash has a window containing the row's
        absolute query position (host-side, O(rows x windows))."""
        out = []
        for i, m in enumerate(meta):
            ws = self.windows.get(m.prompt_hash)
            if ws and any(a <= m.query_position <= b for a, b in ws):
                out.append(i)
        return out

    def sampled_prefill_rows(self, meta: Sequence[RowMeta], n: int) -> list[int]:
        """Row indices kept in SAMPLED mode: query_position % sample_every == 0,
        plus the final sample_tail rows of the chunk (host-side, O(n))."""
        first_tail = max(0, n - self.sample_tail)
        every = self.sample_every
        return [
            i for i, m in enumerate(meta)
            if i >= first_tail or (m.query_position % every == 0)
        ]

    # ---- lifecycle ------------------------------------------------------- #

    def flush(self) -> None:
        """Block until every queued slot has been written (test / step-end use)."""
        while True:
            with self.ring._lock:
                pending = list(self.ring._inflight)
            if not pending or self.fatal is not None:
                break
            pending[-1].copy_event.synchronize()
            time.sleep(0.001)
        if self.fatal is not None:
            raise TraceContractError("trace writer failed") from self.fatal

    def close(self) -> None:
        self.ring.queued.put(None)
        self.writer.join()
        # Flush any ledger edges added after the last capture.
        tail = self.ledger.drain_pending()
        if tail:
            self.ledger_fd.write(PrefixLedger.edges_to_bytes(tail))
        for fd in (self.records_fd, self.ledger_fd):
            fd.flush()
            os.fsync(fd.fileno())
            fd.close()
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "run_id": self.run_id,
            "tp_rank": self.tp_rank,
            "tp_world_size": self.tp_world_size,
            "k": self.k,
            "record_size": record_size(self.k),
            "records_written": self.writer.records_written,
            "bytes_written": self.writer.bytes_written,
            "captures": self.captures,
            "ring_stalls": self.ring.stalls,
            "ring_occupancy": self.ring.occupancy(),
            "sample_mode": self.sample_mode.name.lower(),
            "sample_every": self.sample_every,
            "sample_tail": self.sample_tail,
            "prefill_rows_seen": self.prefill_rows_seen,
            "windows": {k: [list(w) for w in v] for k, v in self.windows.items()},
            "window_rows_seen": self.window_rows_seen,
            "window_rows_captured": self.window_rows_captured,
            "prefill_rows_captured": (
                self.prefill_rows_captured
                if self.sample_mode != SampleMode.FULL else None
            ),
            "ledger_nodes": len(self.ledger.nodes),
            "started_unix": self._started,
            "finished_unix": time.time(),
            "fatal": repr(self.fatal) if self.fatal else None,
            "request_keys": {str(k): v for k, v in self.request_keys.items()},
            **self.manifest_extra,
        }
        with open(self.manifest_path, "w") as f:
            json.dump(manifest, f, indent=2, sort_keys=True)
        if self.fatal is not None:
            raise TraceContractError("trace writer failed") from self.fatal


# --------------------------------------------------------------------------- #
# Host-side helpers used by the model runner and the indexer call sites
# --------------------------------------------------------------------------- #


def request_key(req_id: str) -> int:
    """FNV-1a 64-bit of the engine request id; manifest maps it back."""
    h = 0xCBF29CE484222325
    for b in req_id.encode("utf-8"):
        h ^= b
        h = (h * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return h


def layer_index_from_name(layer_name: str) -> int:
    """'model.layers.17.self_attn.indexer.k_cache' -> 17 (first int component)."""
    for part in layer_name.split("."):
        if part.isdigit():
            return int(part)
    raise ValueError(f"no layer index in {layer_name!r}")


_PLACEHOLDER_MSG = (
    "tollbooth: request {rid} has a placeholder token id (-1) in the CPU token "
    "array; the trace needs exact ids at row-build time, so async scheduling must "
    "be disabled (LLM(..., async_scheduling=False))"
)


class RowBuilder:
    """Turns the model runner's per-forward batch layout into ``RowMeta`` rows.

    Keeps, per request, the ledger node reached at ``num_computed_tokens`` so a
    decode step costs one ledger append per new token instead of re-walking the
    prefix. A request whose computed-token count went backwards (preemption,
    recompute) gets ``attempt_id += 1`` and its prefix is re-interned from the
    token ids, which yields the same nodes because interning is exact.
    """

    def __init__(self, ledger: PrefixLedger, request_keys: dict[int, str]) -> None:
        self.ledger = ledger
        self.request_keys = request_keys
        self._cache: dict[str, tuple[int, int, int]] = {}  # req_id -> (covered, node, attempt)
        self._prompt_hash: dict[str, str] = {}
        self.prompt_hashes: dict[int, str] = {}  # request_key -> prompt hash (for the manifest)

    def forget(self, req_id: str) -> None:
        self._cache.pop(req_id, None)
        self._prompt_hash.pop(req_id, None)

    def build(
        self,
        req_ids: Sequence[str],
        num_scheduled: Sequence[int],
        num_computed: Sequence[int],
        token_ids_cpu,
        decode_threshold: int = 1,
        num_prompt_tokens: Sequence[int] | None = None,
    ) -> list[RowMeta]:
        """``req_ids[i]`` owns rows ``token_ids_cpu[i, c:c+s]`` in batch order,
        where ``c = num_computed[i]`` and ``s = num_scheduled[i]``."""
        rows: list[RowMeta] = []
        for i, rid in enumerate(req_ids):
            s = int(num_scheduled[i])
            if s <= 0:
                continue
            c = int(num_computed[i])
            covered, node, attempt = self._cache.get(rid, (0, 0, 0))
            if covered != c:
                if covered > c:
                    attempt += 1
                prefix = token_ids_cpu[i, :c]
                if c > 0 and int(prefix.min()) < 0:
                    raise TraceContractError(_PLACEHOLDER_MSG.format(rid=rid))
                node = self.ledger.add(prefix) if c > 0 else 0
            key = request_key(rid)
            self.request_keys.setdefault(key, rid)
            ph = self._prompt_hash.get(rid, "")
            if not ph and num_prompt_tokens is not None:
                ph = prompt_hash_of(token_ids_cpu[i, : int(num_prompt_tokens[i])])
                self._prompt_hash[rid] = ph
                self.prompt_hashes[key] = ph
            toks = token_ids_cpu[i, c : c + s]
            if int(toks.min()) < 0:
                raise TraceContractError(_PLACEHOLDER_MSG.format(rid=rid))
            path = self.ledger.add_path_from(node, toks)
            phase = Phase.DECODE if s <= decode_threshold else Phase.PREFILL
            for j in range(s):
                rows.append(RowMeta(key, path[j], c + j, int(toks[j]), attempt, phase, ph))
            self._cache[rid] = (c + s, path[-1], attempt)
        return rows


_ACTIVE: tuple[TraceSession, TraceContext] | None = None


def set_active(session: TraceSession, ctx: TraceContext) -> None:
    global _ACTIVE
    _ACTIVE = (session, ctx)


def clear_active() -> None:
    global _ACTIVE
    _ACTIVE = None


def get_active() -> tuple[TraceSession, TraceContext] | None:
    """Layer-path lookup; None outside a traced forward (dummy runs, profiling,
    graph capture) so every call site degrades to a no-op."""
    return _ACTIVE


def armed() -> bool:
    """True when TOLLBOOTH_DIR is set, i.e. session_from_env would open a session."""
    return bool(os.environ.get("TOLLBOOTH_DIR"))


def session_from_env(
    tp_rank: int,
    tp_world_size: int,
    k: int,
    device: torch.device | str,
    capacity_rows: int,
    manifest_extra: dict | None = None,
) -> TraceSession | None:
    """Build a session from TOLLBOOTH_* environment variables, or None."""
    out_dir = os.environ.get("TOLLBOOTH_DIR")
    if not out_dir:
        return None
    run_id = int(os.environ.get("TOLLBOOTH_RUN_ID", str(int(time.time()))))
    ring_slots = int(os.environ.get("TOLLBOOTH_RING_SLOTS", "64"))
    cap = int(os.environ.get("TOLLBOOTH_CAPACITY_ROWS", str(capacity_rows)))
    mode = SampleMode.parse(os.environ.get("TOLLBOOTH_SAMPLE", "full"))
    windows = None
    wpath = os.environ.get("TOLLBOOTH_WINDOWS")
    if wpath:
        with open(wpath) as f:
            windows = json.load(f)
        mode = SampleMode.WINDOWED
    every = int(os.environ.get("TOLLBOOTH_SAMPLE_EVERY", "64"))
    tail = int(os.environ.get("TOLLBOOTH_SAMPLE_TAIL", "256"))
    return TraceSession(
        out_dir,
        run_id=run_id,
        tp_rank=tp_rank,
        tp_world_size=tp_world_size,
        k=k,
        device=device,
        ring_slots=ring_slots,
        capacity_rows=cap,
        manifest_extra=manifest_extra,
        sample_mode=mode,
        sample_every=every,
        sample_tail=tail,
        windows=windows,
    )


def capture_prefill_chunk(
    trace: tuple[TraceSession, TraceContext],
    layer_id: int,
    logits: torch.Tensor,
    topk_indices: torch.Tensor,
    chunk,
    score_source: int = ScoreSource.SELECTOR,
) -> None:
    """Call after the chunk's top-k (and any DCP merge). ``chunk`` is a
    DeepseekV32IndexerPrefillChunkMetadata: rows are token rows
    ``[token_start, token_end)``; ``cu_seqlen_ks/ke`` are per-row column bounds
    of the causal window inside ``logits``."""
    session, ctx = trace
    n = chunk.token_end - chunk.token_start
    ks = chunk.cu_seqlen_ks[:n]
    ke = chunk.cu_seqlen_ke[:n]
    session.capture(
        ctx,
        layer_id,
        logits,
        topk_indices,
        row_starts=ks,
        prefix_lens=ke - ks,
        rows=slice(chunk.token_start, chunk.token_end),
        phase=Phase.PREFILL,
        score_source=score_source,
    )


def capture_decode(
    trace: tuple[TraceSession, TraceContext],
    layer_id: int,
    logits: torch.Tensor,
    topk_indices: torch.Tensor,
    seq_lens: torch.Tensor,
    num_decode_tokens: int,
    requires_padding: bool,
) -> None:
    """Call after the decode selector dispatch (and any DCP merge), before any
    unpack. Decode rows are the first ``num_decode_tokens`` token rows of the
    forward. Decode indices are request-local already, so row_starts = 0 and
    prefix_lens = seq_lens (per row; 1-D (B,) or 2-D (B, next_n) flattened)."""
    if num_decode_tokens <= 0:
        return
    if requires_padding:
        raise NotImplementedError(
            "tollbooth: padded decode rows (uneven decode_lens, i.e. speculative "
            "decoding or short chunked prefills routed as decode) are outside the "
            "traced configuration; run with next_n == 1"
        )
    session, ctx = trace
    n = num_decode_tokens
    prefix = seq_lens.reshape(-1)[:n]
    session.capture(
        ctx,
        layer_id,
        logits[:n],
        topk_indices[:n],
        row_starts=torch.zeros(n, dtype=torch.int32, device=logits.device),
        prefix_lens=prefix,
        rows=slice(0, n),
        phase=Phase.DECODE,
        score_source=ScoreSource.SELECTOR,
    )


# --------------------------------------------------------------------------- #
# Validation helper (offline and test use only; never on the layer path)
# --------------------------------------------------------------------------- #


def validate_topk_row(
    window_scores, ids, scores, tau: float, k: int
) -> list[str]:
    """Check one record against the live scores of its causal window.

    ``window_scores``: f32 [prefix_len] scores of every key the selector saw;
    ``ids``/``scores``: the record's int32[k] / f32[k]; ``tau``: the record's
    tau. Returns a list of problems (empty == valid). Enforces, when
    prefix_len >= k:
      (a) selected score multiset == true top-k score multiset,
      (b) every id scoring strictly above tau is selected,
      (c) #selected strictly above tau + #selected at tau == k,
      (d) ids unique and in range, scores == window[ids] bitwise,
          min(selected) == tau.
    Any choice among ids tied at tau is accepted; [10, 9, 9] with k=2 selecting
    both 9s is rejected by (a) and (b).
    """
    problems: list[str] = []
    w = np.asarray(window_scores, dtype=np.float32)
    ids = np.asarray(ids, dtype=np.int32)
    sc = np.asarray(scores, dtype=np.float32)
    prefix_len = int(w.shape[0])
    valid = ids >= 0
    sel = ids[valid]
    ssc = sc[valid]
    if not np.all(np.isnan(sc[~valid])):
        problems.append("non-NaN score at a -1 slot")
    if len(np.unique(sel)) != len(sel):
        problems.append("duplicate ids")
    if len(sel) and (sel.min() < 0 or sel.max() >= prefix_len):
        problems.append("id outside prefix")
        return problems
    if not np.array_equal(ssc.view(np.uint32), w[sel].view(np.uint32)):
        problems.append("score != window score at id (bitwise)")
    expect_n = min(k, prefix_len)
    if len(sel) != expect_n:
        problems.append(f"valid_count {len(sel)} != min(k, prefix_len) {expect_n}")
        return problems
    if prefix_len < k:
        if not math.isnan(tau):
            problems.append("tau must be NaN for a prefix shorter than k")
        if set(sel.tolist()) != set(range(prefix_len)):
            problems.append("short prefix must select every position")
        return problems
    top = np.sort(w)[::-1][:k]
    if not np.array_equal(np.sort(ssc)[::-1].view(np.uint32), top.view(np.uint32)):
        problems.append("(a) selected score multiset != true top-k multiset")
    if math.isnan(tau) or np.float32(tau).view(np.uint32) != ssc.min().view(np.uint32):
        problems.append("(d) tau != min(selected scores)")
        return problems
    t = np.float32(tau)
    above = np.nonzero(w > t)[0]
    if not np.isin(above, sel).all():
        problems.append("(b) an id scoring strictly above tau is not selected")
    n_above = int((ssc > t).sum())
    n_at = int((ssc == t).sum())
    if n_above + n_at != k:
        problems.append("(c) #above tau + #at tau != k")
    if n_above != len(above):
        problems.append("(b) #selected above tau != #window above tau")
    return problems
