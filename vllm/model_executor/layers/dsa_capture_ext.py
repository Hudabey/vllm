# SPDX-License-Identifier: Apache-2.0
"""tollbooth capture extensions C2 (analysis/CAPTURE_BRIEF.md): three streams that
make the exact-search bound evaluable offline, alongside the existing selection
records of ``dsa_trace``.

  S2  queries + head weights   per captured row per layer: the FP8 query block
                               ``q_quant[row]`` (64 x 128 e4m3, 8,192 B) and the
                               fp32 folded head weights ``weights[row]`` (64 x 4 B)
                               exactly as the logits kernel receives them, behind
                               the same 64-byte header as the selection records.
  S3  key-cache pages          host-side, after the forward: snapshot A of every
                               indexer-cache page of a request at its first captured
                               row (page-verbatim, 64 positions x 128 e4m3 values then
                               64 fp32 scales = 8,448 B, with the logical->physical
                               map), the 132-byte appended key of every captured
                               decode row, and snapshot B (page hashes + the window's
                               pages verbatim) at the request's last captured row.
  S4  full score rows          for an explicit list of (prompt hash, position): the
                               fp32 logits row ``logits[i, start:start+P]`` cloned ON
                               DEVICE at capture time (the logits buffer is reused by
                               the next layer), copied to pinned memory on the copy
                               stream, written once. One retained copy per row.

Layer-path rules are those of ``dsa_trace``: device ops only (index_select, view,
clone, copy_ non_blocking, event.record); no .item()/.cpu()/file I/O; never
touches the selector's buffers except to read. S3 runs after the forward on the
host and does synchronous D2H copies by design (documented in the brief).

Armed by environment, in addition to TOLLBOOTH_DIR:
  TOLLBOOTH_C2=1           enable S2 for every row the base session captures
  TOLLBOOTH_PAGES=1        enable S3 (needs the model runner's post_forward call)
  TOLLBOOTH_ROWS=<path>    enable S4: JSON {prompt_hash: [positions, ...]}
"""

from __future__ import annotations

import hashlib
import json
import os
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import torch

try:  # inside the vllm package
    from . import dsa_trace
except ImportError:  # loaded by file path in the CPU test suite
    import dsa_trace  # type: ignore[no-redef]

N_HEADS = 64
HEAD_DIM = 128
Q_BYTES = N_HEADS * HEAD_DIM  # 8,192: one e4m3 byte per element
W_BYTES = N_HEADS * 4  # 256
HEADER_SIZE = dsa_trace.HEADER_DTYPE.itemsize  # 64
QUERY_RECORD_SIZE = HEADER_SIZE + Q_BYTES + W_BYTES  # 8,512
KEY_BYTES = HEAD_DIM + 4  # 132: values + fp32 scale
C2_SCHEMA_VERSION = 1


def query_record_dtype() -> np.dtype:
    return np.dtype(
        [
            ("header", dsa_trace.HEADER_DTYPE),
            ("q", np.uint8, (Q_BYTES,)),
            ("w", np.float32, (N_HEADS,)),
        ]
    )


def read_query_records(path: str) -> np.ndarray:
    with open(path, "rb") as f:
        return np.frombuffer(f.read(), dtype=query_record_dtype()).copy()


# --------------------------------------------------------------------------- #
# Pinned ring for the layer-path streams (S2, S4)
# --------------------------------------------------------------------------- #


@dataclass
class ExtSlot:
    index: int
    buf: torch.Tensor  # uint8 [capacity_bytes], pinned on CUDA
    producer_event: dsa_trace.EventLike
    copy_event: dsa_trace.EventLike
    kind: str = ""
    nbytes: int = 0
    rows: int = 0
    meta: Sequence[dsa_trace.RowMeta] = ()
    step_id: int = 0
    layer_id: int = 0
    phase: int = dsa_trace.Phase.PREFILL
    row_info: dict = field(default_factory=dict)  # S4: prompt_hash, position, P
    device_refs: tuple = ()
    state: int = dsa_trace.SlotState.FREE


class ExtRing:
    """Same contract as ``dsa_trace.TraceRing``: never drops, backpressure blocks
    on an event, writer failure releases queued slots and makes acquire raise."""

    def __init__(self, num_slots: int, capacity_bytes: int, device: torch.device,
                 pin_memory: bool | None = None) -> None:
        if num_slots < 1 or capacity_bytes < QUERY_RECORD_SIZE:
            raise ValueError("ring needs >= 1 slot and room for one record")
        self.device = device
        self.capacity_bytes = capacity_bytes
        pin = (device.type == "cuda") if pin_memory is None else pin_memory
        self.slots = [
            ExtSlot(i, torch.empty((capacity_bytes,), dtype=torch.uint8, pin_memory=pin),
                    dsa_trace._make_event(device), dsa_trace._make_event(device))
            for i in range(num_slots)
        ]
        self._free: queue.SimpleQueue[ExtSlot] = queue.SimpleQueue()
        for s in self.slots:
            self._free.put(s)
        self.queued: queue.Queue[ExtSlot | None] = queue.Queue()
        self._inflight: list[ExtSlot] = []
        self._lock = threading.Lock()
        self.stalls = 0
        self.failed: BaseException | None = None
        self.copy_stream = torch.cuda.Stream() if device.type == "cuda" else None

    def fail(self, exc: BaseException) -> None:
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
            raise dsa_trace.TraceContractError("c2 writer failed; ring is closed") from self.failed

    def acquire(self, nbytes: int) -> ExtSlot:
        if nbytes > self.capacity_bytes:
            raise ValueError(f"{nbytes} bytes exceed slot capacity {self.capacity_bytes}")
        self._raise_if_failed()
        try:
            slot = self._free.get_nowait()
        except queue.Empty:
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
        slot.state = dsa_trace.SlotState.FILLING
        slot.nbytes = nbytes
        return slot

    def submit(self, slot: ExtSlot, copies: Sequence[tuple[torch.Tensor, torch.Tensor]],
               device_refs: tuple) -> None:
        """``copies`` = (pinned destination view, device source) pairs."""
        slot.device_refs = device_refs
        if self.copy_stream is not None:
            cur = torch.cuda.current_stream(self.device)
            slot.producer_event.record(cur)
            with torch.cuda.stream(self.copy_stream):
                self.copy_stream.wait_event(slot.producer_event)
                for dst, src in copies:
                    dst.copy_(src, non_blocking=True)
                slot.copy_event.record(self.copy_stream)
        else:
            slot.producer_event.record()
            for dst, src in copies:
                dst.copy_(src)
            slot.copy_event.record()
        slot.state = dsa_trace.SlotState.QUEUED
        with self._lock:
            self._inflight.append(slot)
        self.queued.put(slot)

    def release(self, slot: ExtSlot) -> None:
        with self._lock:
            if slot in self._inflight:
                self._inflight.remove(slot)
        slot.device_refs = ()
        slot.meta = ()
        slot.row_info = {}
        slot.kind = ""
        slot.rows = 0
        slot.nbytes = 0
        slot.state = dsa_trace.SlotState.FREE
        self._free.put(slot)


# --------------------------------------------------------------------------- #
# Serialization
# --------------------------------------------------------------------------- #


def _header(meta: Sequence[dsa_trace.RowMeta], run_id: int, step_id: int, layer_id: int,
            tp_rank: int, phase: int, prefix_lens: np.ndarray, sample_mode: int) -> np.ndarray:
    hdr = np.zeros(len(meta), dtype=dsa_trace.HEADER_DTYPE)
    hdr["run_id"] = run_id
    hdr["step_id"] = step_id
    hdr["request_key"] = [m.request_key for m in meta]
    hdr["prefix_node_id"] = [m.prefix_node_id for m in meta]
    hdr["query_position"] = [m.query_position for m in meta]
    hdr["query_token_id"] = [m.query_token_id for m in meta]
    hdr["attempt_id"] = [m.attempt_id for m in meta]
    hdr["layer_id"] = layer_id
    hdr["tp_rank"] = tp_rank
    hdr["prefix_length"] = prefix_lens.astype(np.uint32)
    hdr["valid_count"] = 0
    hdr["tau"] = np.nan
    flags = np.zeros(len(meta), dtype=np.uint16)
    if phase == dsa_trace.Phase.DECODE:
        flags |= np.uint16(dsa_trace.Flag.DECODE)
    flags |= np.uint16(int(sample_mode) << dsa_trace.SAMPLE_SHIFT)
    hdr["flags"] = flags
    hdr["reserved"] = C2_SCHEMA_VERSION
    return hdr


def serialize_queries(slot: ExtSlot, run_id: int, tp_rank: int, sample_mode: int) -> bytes:
    n = slot.rows
    rec = np.zeros(n, dtype=query_record_dtype())
    rec["header"] = _header(slot.meta, run_id, slot.step_id, slot.layer_id, tp_rank, slot.phase,
                            np.asarray(slot.row_info["prefix_lens"]), sample_mode)
    nq = n * Q_BYTES
    rec["q"] = slot.buf[:nq].numpy().reshape(n, Q_BYTES)
    rec["w"] = slot.buf[nq : nq + n * W_BYTES].numpy().view(np.float32).reshape(n, N_HEADS)
    return rec.tobytes()


# --------------------------------------------------------------------------- #
# Writer thread
# --------------------------------------------------------------------------- #


class ExtWriter(threading.Thread):
    def __init__(self, session: "C2Session") -> None:
        super().__init__(name=f"tollbooth-c2-writer-r{session.tp_rank}", daemon=True)
        self.s = session

    def run(self) -> None:
        ring = self.s.ring
        while True:
            slot = ring.queued.get()
            if slot is None:
                break
            try:
                slot.state = dsa_trace.SlotState.WRITING
                slot.copy_event.synchronize()
                if slot.kind == "queries":
                    data = serialize_queries(slot, self.s.run_id, self.s.tp_rank, self.s.sample_mode)
                    self.s.queries_fd.write(data)
                    self.s.queries_fd.flush()
                    self.s.stats["queries_records"] += slot.rows
                    self.s.stats["queries_bytes"] += len(data)
                elif slot.kind == "row":
                    info = slot.row_info
                    data = slot.buf[: slot.nbytes].numpy().tobytes()  # copy out of pinned memory
                    name = f"{info['prompt_hash']}.p{info['position']}.L{slot.layer_id}.f32"
                    path = os.path.join(self.s.rows_dir, name)
                    if os.path.exists(path):
                        raise dsa_trace.TraceContractError(f"c2: full row written twice: {name}")
                    with open(path, "wb") as f:
                        f.write(data)
                    self.s.rows_written.append({
                        "file": os.path.join("rows", name), "prompt_hash": info["prompt_hash"],
                        "position": info["position"], "layer_id": slot.layer_id,
                        "prefix_length": info["P"], "bytes": len(data),
                        "sha256": hashlib.sha256(data).hexdigest(), "step_id": slot.step_id,
                        "phase": "decode" if slot.phase == dsa_trace.Phase.DECODE else "prefill",
                        "request_key": int(slot.meta[0].request_key),
                    })
                    self.s.stats["rows_written"] += 1
                    self.s.stats["rows_bytes"] += len(data)
                else:
                    raise dsa_trace.TraceContractError(f"c2: unknown slot kind {slot.kind!r}")
            except BaseException as e:  # noqa: BLE001
                self.s.fatal = e
                ring.release(slot)
                ring.fail(e)
                break
            ring.release(slot)


# --------------------------------------------------------------------------- #
# Session
# --------------------------------------------------------------------------- #


class C2Session:
    """Owns the S2/S4 ring + writer and the S3 host-side snapshotter for one rank.
    Attached to the base ``TraceSession`` as ``session.c2`` so the call sites in
    ``sparse_attn_indexer`` find it through ``dsa_trace.get_active()``."""

    def __init__(
        self,
        base: dsa_trace.TraceSession,
        device: torch.device | str = "cpu",
        max_model_len: int = 163840,
        queries: bool = True,
        rows_spec: dict[str, set[int]] | None = None,
        pages: bool = False,
        num_slots: int = 32,
        capacity_bytes: int | None = None,
        pin_memory: bool | None = None,
        page_bytes: int = 64 * KEY_BYTES,
        block_size: int = 64,
        layers: set[int] | None = None,
        pages_hash_only: bool = False,
    ) -> None:
        self.base = base
        self.out_dir = base.out_dir
        self.run_id = base.run_id
        self.tp_rank = base.tp_rank
        self.sample_mode = int(base.sample_mode)
        self.device = torch.device(device)
        self.enable_queries = bool(queries)
        self.rows_spec: dict[str, set[int]] = {k: set(int(p) for p in v) for k, v in (rows_spec or {}).items()}
        self.enable_pages = bool(pages)
        self.block_size = int(block_size)
        self.page_bytes = int(page_bytes)
        self.layers: set[int] | None = None if layers is None else {int(l) for l in layers}  # S2/S4 layer filter
        self.pages_hash_only = bool(pages_hash_only)  # S3: hashes only, no page bytes (rank-1 spot check)
        self.max_model_len = int(max_model_len)
        cap = capacity_bytes or max(8 << 20, max_model_len * 4)
        self.ring = ExtRing(num_slots, cap, self.device, pin_memory)
        self.fatal: BaseException | None = None
        self.stats = {"queries_records": 0, "queries_bytes": 0, "rows_written": 0, "rows_bytes": 0,
                      "rows_requested_seen": 0, "pages_bytes": 0, "pages_d2h_bytes": 0,
                      "appended_keys": 0, "snapshots_a": 0, "snapshots_b": 0}
        self.rows_written: list[dict] = []
        self.pages_manifest: dict[str, dict] = {}
        self._pages_a_done: set[str] = set()
        self._pages_b_done: set[str] = set()
        self.rows_dir = os.path.join(self.out_dir, "rows")
        self.pages_dir = os.path.join(self.out_dir, "pages")
        os.makedirs(self.rows_dir, exist_ok=True)
        os.makedirs(self.pages_dir, exist_ok=True)
        self.queries_path = os.path.join(self.out_dir, f"tollbooth.rank{self.tp_rank}.queries")
        self.manifest_path = os.path.join(self.out_dir, f"tollbooth.rank{self.tp_rank}.c2.json")
        self.queries_fd = open(self.queries_path, "ab")
        self.manifest_extra: dict = {}
        self._started = time.time()
        self.writer = ExtWriter(self)
        self.writer.start()
        base.c2 = self  # discovered by the call sites

    # ---- layer path ----------------------------------------------------- #

    def _check(self) -> None:
        if self.fatal is not None:
            raise dsa_trace.TraceContractError("c2 writer failed earlier") from self.fatal

    def capture_queries(
        self,
        ctx: dsa_trace.TraceContext,
        layer_id: int,
        q_quant: torch.Tensor,  # [n, N_HEADS, HEAD_DIM] float8 (or uint8 view) for the kept rows
        weights: torch.Tensor,  # [n, N_HEADS] fp32 for the kept rows
        meta: Sequence[dsa_trace.RowMeta],
        prefix_lens: torch.Tensor,  # int32 [n]
        phase: int,
    ) -> None:
        if not self.enable_queries or (self.layers is not None and int(layer_id) not in self.layers):
            return
        self._check()
        n = len(meta)
        if n == 0:
            return
        if q_quant.shape[0] != n or weights.shape[0] != n:
            raise ValueError("c2: query/weight rows do not match row metas")
        if q_quant.dtype != torch.uint8:
            q_quant = q_quant.view(torch.uint8)
        qb = q_quant.reshape(n, Q_BYTES)
        if qb.shape[1] != Q_BYTES:
            raise ValueError(f"c2: query block is {qb.shape[1]} bytes, expected {Q_BYTES}")
        w = weights.to(torch.float32).reshape(n, N_HEADS)
        # prefix_lens are needed by the writer thread for the header; they are tiny
        # and the base capture already made them host-visible only through the
        # ring, so carry them the same way: append them to the slot payload.
        pl = prefix_lens.to(torch.int32).reshape(n)
        per_rec = QUERY_RECORD_SIZE - HEADER_SIZE + 4
        rows_per_slot = max(1, self.ring.capacity_bytes // per_rec)
        for start in range(0, n, rows_per_slot):
            end = min(start + rows_per_slot, n)
            m = end - start
            nq, nw = m * Q_BYTES, m * W_BYTES
            slot = self.ring.acquire(nq + nw + m * 4)
            slot.kind = "queries"
            slot.rows = m
            slot.meta = list(meta[start:end])
            slot.step_id = ctx.step_id
            slot.layer_id = layer_id
            slot.phase = phase
            q_src = qb[start:end].contiguous()
            w_src = w[start:end].contiguous()
            pl_src = pl[start:end].contiguous()
            copies = (
                (slot.buf[:nq].view(m, Q_BYTES), q_src),
                (slot.buf[nq : nq + nw].view(torch.float32).view(m, N_HEADS), w_src),
                (slot.buf[nq + nw : nq + nw + m * 4].view(torch.int32), pl_src),
            )
            slot.row_info = {"prefix_lens": _PinnedInts(slot.buf, nq + nw, m)}
            self.ring.submit(slot, copies, (q_src, w_src, pl_src))
        self.stats["queries_records_submitted"] = self.stats.get("queries_records_submitted", 0) + n

    def wanted_rows(self, meta: Sequence[dsa_trace.RowMeta]) -> list[int]:
        if not self.rows_spec:
            return []
        out = []
        for i, m in enumerate(meta):
            s = self.rows_spec.get(m.prompt_hash)
            if s and m.query_position in s:
                out.append(i)
        return out

    def capture_full_rows(
        self,
        ctx: dsa_trace.TraceContext,
        layer_id: int,
        logits: torch.Tensor,  # f32 [n, C] live buffer for the kept rows (read only)
        row_starts: torch.Tensor,  # int32 [n]
        prefix_lens: torch.Tensor,  # int32 [n]
        meta: Sequence[dsa_trace.RowMeta],
        phase: int,
        starts_cpu: Sequence[int] | None = None,
        prefix_cpu: Sequence[int] | None = None,
    ) -> None:
        """Clone the requested rows ON DEVICE now: ``logits`` is a buffer the next
        layer (or the next chunk) overwrites, so a view would be clobbered before
        the copy stream reads it. ``starts_cpu``/``prefix_cpu`` are the host-side
        values of ``row_starts``/``prefix_lens`` when the caller has them (the
        prefill chunk metadata does); without them one bounded .tolist() is done
        for the wanted rows only (never for rows nobody asked for)."""
        if not self.rows_spec or (self.layers is not None and int(layer_id) not in self.layers):
            return
        self._check()
        want = self.wanted_rows(meta)
        if not want:
            return
        self.stats["rows_requested_seen"] += len(want)
        if starts_cpu is None or prefix_cpu is None:
            idx = torch.tensor(want, dtype=torch.int64, device=logits.device)
            starts_l = row_starts.index_select(0, idx).tolist()
            prefix_l = prefix_lens.index_select(0, idx).tolist()
        else:
            starts_l = [int(starts_cpu[i]) for i in want]
            prefix_l = [int(prefix_cpu[i]) for i in want]
        for j, i in enumerate(want):
            rs, P = int(starts_l[j]), int(prefix_l[j])
            if P <= 0:
                continue
            if rs + P > logits.shape[1]:
                raise dsa_trace.TraceContractError(
                    f"c2: row {meta[i].query_position} needs columns {rs}..{rs + P} of {logits.shape[1]}")
            row = logits[i, rs : rs + P].clone()  # owned device copy, on the current stream
            nbytes = P * 4
            slot = self.ring.acquire(nbytes)
            slot.kind = "row"
            slot.rows = 1
            slot.meta = [meta[i]]
            slot.step_id = ctx.step_id
            slot.layer_id = layer_id
            slot.phase = phase
            slot.row_info = {"prompt_hash": meta[i].prompt_hash, "position": int(meta[i].query_position), "P": P}
            self.ring.submit(slot, ((slot.buf[:nbytes].view(torch.float32), row),), (row,))

    # ---- host side (after the forward) ---------------------------------- #

    def post_forward(self, ctx: dsa_trace.TraceContext, cache_access: "CacheAccess") -> None:
        """S3. ``cache_access`` abstracts the model runner (block tables, per-layer
        cache tensors, request index by request key). Synchronous D2H by design."""
        if not self.enable_pages:
            return
        self._check()
        rows = ctx.rows
        kept = self.base.windowed_rows(rows) if self.base.sample_mode == dsa_trace.SampleMode.WINDOWED else list(range(len(rows)))
        if not kept:
            return
        by_hash: dict[str, list[dsa_trace.RowMeta]] = {}
        for i in kept:
            by_hash.setdefault(rows[i].prompt_hash, []).append(rows[i])
        for ph, metas in by_hash.items():
            key = metas[0].request_key
            npos = max(m.query_position for m in rows if m.request_key == key) + 1
            bt = np.asarray(cache_access.block_table_row(key))
            nblocks = -(-npos // self.block_size)
            if nblocks > len(bt):
                raise dsa_trace.TraceContractError(f"c2: request {ph[:8]} needs {nblocks} blocks, table has {len(bt)}")
            phys = bt[:nblocks].astype(np.int64)
            entry = self.pages_manifest.setdefault(ph, {"request_key": int(key), "layers": {}, "appended": [],
                                                        "block_size": self.block_size, "page_bytes": self.page_bytes})
            if ph not in self._pages_a_done:
                self._snapshot(ph, phys, npos, cache_access, entry, "A", ctx.step_id)
                self._pages_a_done.add(ph)
            # appended keys for captured decode rows
            for m in metas:
                if m.phase != dsa_trace.Phase.DECODE:
                    continue
                p = int(m.query_position)
                blk, off = int(bt[p // self.block_size]), p % self.block_size
                keys = []
                for layer_id, kv2d in cache_access.layer_caches():
                    page = kv2d[blk]
                    vals = page[off * HEAD_DIM : (off + 1) * HEAD_DIM]
                    scale = page[self.block_size * HEAD_DIM + off * 4 : self.block_size * HEAD_DIM + (off + 1) * 4]
                    keys.append((layer_id, torch.cat([vals, scale]).cpu().numpy().tobytes()))
                data = b"".join(k for _, k in keys)
                fn = f"{ph}.appended.records"
                with open(os.path.join(self.pages_dir, fn), "ab") as f:
                    hdr = np.zeros(1, dtype=dsa_trace.HEADER_DTYPE)
                    hdr["run_id"] = self.run_id; hdr["step_id"] = ctx.step_id; hdr["request_key"] = m.request_key
                    hdr["prefix_node_id"] = m.prefix_node_id; hdr["query_position"] = p; hdr["query_token_id"] = m.query_token_id
                    hdr["attempt_id"] = m.attempt_id; hdr["layer_id"] = len(keys); hdr["tp_rank"] = self.tp_rank
                    hdr["prefix_length"] = p + 1; hdr["flags"] = int(dsa_trace.Flag.DECODE); hdr["tau"] = np.nan
                    hdr["reserved"] = C2_SCHEMA_VERSION
                    f.write(hdr.tobytes() + data)
                entry["appended"].append({"position": p, "physical_page": blk, "offset": off,
                                          "layers": [l for l, _ in keys], "bytes": HEADER_SIZE + len(data)})
                self.stats["appended_keys"] += 1
                self.stats["pages_bytes"] += HEADER_SIZE + len(data)
            # snapshot B at the request's last captured position (window end)
            ends = [b for a, b in self.base.windows.get(ph, [])]
            if ends and ph not in self._pages_b_done and any(m.query_position == max(ends) for m in metas):
                self._snapshot(ph, phys, npos, cache_access, entry, "B", ctx.step_id)
                self._pages_b_done.add(ph)

    def _snapshot(self, ph: str, phys: np.ndarray, npos: int, cache_access: "CacheAccess",
                  entry: dict, which: str, step_id: int) -> None:
        wins = self.base.windows.get(ph, [])
        lo = min((a for a, b in wins), default=0)
        hi = max((b for a, b in wins), default=npos - 1)
        win_blocks = set(range(lo // self.block_size, hi // self.block_size + 1))
        for layer_id, kv2d in cache_access.layer_caches():
            if kv2d.shape[1] != self.page_bytes:
                raise dsa_trace.TraceContractError(f"c2: page is {kv2d.shape[1]} bytes, expected {self.page_bytes}")
            idx = torch.as_tensor(phys, dtype=torch.int64, device=kv2d.device)
            pages = kv2d.index_select(0, idx).cpu().numpy()  # synchronous D2H, host side
            self.stats["pages_d2h_bytes"] += int(pages.nbytes)
            hashes = [hashlib.sha256(pages[i].tobytes()).hexdigest() for i in range(len(phys))]
            lay = entry["layers"].setdefault(str(layer_id), {})
            if which == "A":
                fn = f"{ph}.L{layer_id}.A.pages"
                if not self.pages_hash_only:
                    with open(os.path.join(self.pages_dir, fn), "wb") as f:
                        f.write(pages.tobytes())
                lay["A"] = {"file": None if self.pages_hash_only else os.path.join("pages", fn),
                            "bytes": 0 if self.pages_hash_only else int(pages.nbytes), "pages": len(phys),
                            "positions_covered": int(npos), "physical_pages": phys.tolist(),
                            "last_page_fill": int(npos - (len(phys) - 1) * self.block_size),
                            "sha256_pages": hashes, "step_id": step_id}
                self.stats["pages_bytes"] += 0 if self.pages_hash_only else int(pages.nbytes)
            else:
                keep = sorted(b for b in win_blocks if b < len(phys))
                fn = f"{ph}.L{layer_id}.B.window.pages"
                data = b"" if self.pages_hash_only else b"".join(pages[b].tobytes() for b in keep)
                if not self.pages_hash_only:
                    with open(os.path.join(self.pages_dir, fn), "wb") as f:
                        f.write(data)
                lay["B"] = {"file": None if self.pages_hash_only else os.path.join("pages", fn), "bytes": len(data), "pages": len(phys),
                            "positions_covered": int(npos), "physical_pages": phys.tolist(),
                            "window_logical_pages": keep, "sha256_pages": hashes, "step_id": step_id}
                self.stats["pages_bytes"] += len(data)
        self.stats["snapshots_a" if which == "A" else "snapshots_b"] += 1

    # ---- lifecycle ------------------------------------------------------- #

    def flush(self) -> None:
        while True:
            with self.ring._lock:
                pending = list(self.ring._inflight)
            if not pending or self.fatal is not None:
                break
            pending[-1].copy_event.synchronize()
            time.sleep(0.001)
        if self.fatal is not None:
            raise dsa_trace.TraceContractError("c2 writer failed") from self.fatal

    def close(self) -> None:
        self.ring.queued.put(None)
        self.writer.join()
        self.queries_fd.flush()
        os.fsync(self.queries_fd.fileno())
        self.queries_fd.close()
        manifest = {
            "c2_schema_version": C2_SCHEMA_VERSION,
            "run_id": self.run_id,
            "tp_rank": self.tp_rank,
            "streams": {"queries": self.enable_queries, "rows": bool(self.rows_spec), "pages": self.enable_pages},
            "query_record_size": QUERY_RECORD_SIZE,
            "key_bytes": KEY_BYTES,
            "page_bytes": self.page_bytes,
            "block_size": self.block_size,
            "page_layout": "64 x 128 e4m3 values, then 64 fp32 scales (indexer_k_quant_and_cache)",
            "rows_spec": {k: sorted(v) for k, v in self.rows_spec.items()},
            "layers": None if self.layers is None else sorted(self.layers),
            "pages_hash_only": self.pages_hash_only,
            "rows": self.rows_written,
            "pages": self.pages_manifest,
            "stats": self.stats,
            "ring_stalls": self.ring.stalls,
            "started_unix": self._started,
            "finished_unix": time.time(),
            "fatal": repr(self.fatal) if self.fatal else None,
            **self.manifest_extra,
        }
        if os.path.exists(self.queries_path):
            manifest["queries_file"] = os.path.basename(self.queries_path)
            manifest["queries_bytes_on_disk"] = os.path.getsize(self.queries_path)
        with open(self.manifest_path, "w") as f:
            json.dump(manifest, f, indent=2, sort_keys=True)
        if self.fatal is not None:
            raise dsa_trace.TraceContractError("c2 writer failed") from self.fatal


class _PinnedInts:
    """Late-bound view of prefix lengths living in the slot's pinned buffer (read
    by the writer thread after the copy event completed)."""

    def __init__(self, buf: torch.Tensor, offset: int, n: int) -> None:
        self.buf, self.offset, self.n = buf, offset, n

    def __array__(self, dtype=None, copy=None):
        a = self.buf[self.offset : self.offset + self.n * 4].numpy().view(np.int32)
        return a.astype(dtype) if dtype is not None else a


# --------------------------------------------------------------------------- #
# Cache access abstraction (model runner or test double)
# --------------------------------------------------------------------------- #


class CacheAccess:
    """What S3 needs from the runner: the request's block-table row (logical ->
    physical page ids, by request key) and every indexer layer's cache as a 2-D
    ``[num_pages, page_bytes]`` uint8 view."""

    def block_table_row(self, request_key: int) -> np.ndarray:  # pragma: no cover - interface
        raise NotImplementedError

    def layer_caches(self) -> list[tuple[int, torch.Tensor]]:  # pragma: no cover - interface
        raise NotImplementedError


class RunnerCacheAccess(CacheAccess):
    """Adapter over ``GPUModelRunner``: finds the kv-cache group that holds the
    indexer caches, the group's block table and each layer's cache tensor."""

    def __init__(self, runner, request_keys: dict[int, str]) -> None:
        self.runner = runner
        self.request_keys = request_keys
        self._layers: list[tuple[int, torch.Tensor]] | None = None
        self._gid: int | None = None

    def _resolve(self) -> None:
        cfg = self.runner.kv_cache_config
        fwd = self.runner.compilation_config.static_forward_context
        for gid, group in enumerate(cfg.kv_cache_groups):
            names = [n for n in group.layer_names if ".indexer." in n]
            if not names:
                continue
            layers = []
            for n in names:
                mod = fwd[n]
                kv = mod.kv_cache
                if isinstance(kv, (list, tuple)):
                    kv = kv[0]
                kv2d = kv.reshape(kv.shape[0], -1)
                layers.append((dsa_trace.layer_index_from_name(n), kv2d))
            layers.sort(key=lambda t: t[0])
            self._layers, self._gid = layers, gid
            return
        raise dsa_trace.TraceContractError("c2: no kv-cache group holds indexer layers")

    def block_table_row(self, request_key: int) -> np.ndarray:
        if self._gid is None:
            self._resolve()
        rid = self.request_keys[request_key]
        ridx = self.runner.input_batch.req_id_to_index[rid]
        return self.runner.input_batch.block_table[self._gid].get_numpy_array()[ridx].copy()

    def layer_caches(self) -> list[tuple[int, torch.Tensor]]:
        if self._layers is None:
            self._resolve()
        assert self._layers is not None
        return self._layers


# --------------------------------------------------------------------------- #
# Environment
# --------------------------------------------------------------------------- #


def c2_from_env(base: dsa_trace.TraceSession, device: torch.device | str, max_model_len: int,
                block_size: int = 64) -> C2Session | None:
    """TOLLBOOTH_C2 / TOLLBOOTH_PAGES / TOLLBOOTH_ROWS arm every rank identically.
    TOLLBOOTH_C2_SPEC=<json> arms per rank instead: {"<tp_rank>": {"queries": bool,
    "pages": bool, "pages_hash_only": bool, "rows": <path or null>, "layers": [..] or null}};
    ranks absent from the spec get no C2 session (the brief's S5: rank 1 = layers 0 and 42
    for S2/S4, page hashes only)."""
    spec_path = os.environ.get("TOLLBOOTH_C2_SPEC")
    if spec_path:
        with open(spec_path) as f:
            spec = json.load(f)
        cfg = spec.get(str(base.tp_rank))
        if not cfg:
            return None
        queries = bool(cfg.get("queries", False)); pages = bool(cfg.get("pages", False))
        rows_path = cfg.get("rows"); layers = cfg.get("layers"); hash_only = bool(cfg.get("pages_hash_only", False))
    else:
        queries = os.environ.get("TOLLBOOTH_C2", "") not in ("", "0", "false")
        pages = os.environ.get("TOLLBOOTH_PAGES", "") not in ("", "0", "false")
        rows_path = os.environ.get("TOLLBOOTH_ROWS"); layers = None; hash_only = False
    rows_spec = None
    if rows_path:
        with open(rows_path) as f:
            raw = json.load(f)
        rows_spec = {k: set(int(p) for p in v) for k, v in raw.items()}
    if not (queries or pages or rows_spec):
        return None
    return C2Session(base, device=device, max_model_len=max_model_len, queries=queries,
                     rows_spec=rows_spec, pages=pages, block_size=block_size,
                     layers=None if layers is None else set(int(l) for l in layers), pages_hash_only=hash_only)


def get_c2(session: dsa_trace.TraceSession) -> C2Session | None:
    return getattr(session, "c2", None)
