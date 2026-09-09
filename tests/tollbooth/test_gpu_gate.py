# SPDX-License-Identifier: Apache-2.0
"""tollbooth single-GPU kernel gate (codex-review "Validation before renting GPUs",
POD_BRIEF section 3).

Everything here runs the *real* kernels: DeepGEMM ``fp8_fp4_mqa_logits`` /
``fp8_fp4_paged_mqa_logits``, vLLM's ``top_k_per_row_prefill`` and the decode
top-k dispatch that ``sparse_attn_indexer`` picks on this GPU, the FP8 indexer
K-cache insert/gather kernels, and the real ``DeepseekV32IndexerMetadataBuilder``.
The forward goes through ``sparse_attn_indexer()`` itself. There are no kernel
mocks; the only stand-ins are (a) a default ``VllmConfig`` whose ``model_config``
is a namespace carrying ``max_model_len`` (the only field the builder reads),
(b) a ``ForwardContext`` built directly, and (c) for the dense-bypass case a
main-MLA metadata namespace with ``prefill.use_dense_mha=True`` (the MLA
metadata is only inspected, never consumed by a kernel). A thin spy wrapper
around the two DeepGEMM logits calls records the live FP32 logits so the dumped
scores can be checked bitwise; it calls the real kernel unchanged.

Facts for REPORT.md are appended to ``$TOLLBOOTH_GATE_OUT/gate-results.json``.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm import _custom_ops as ops
from vllm.config import CUDAGraphMode, VllmConfig
from vllm.forward_context import ForwardContext, override_forward_context
from vllm.model_executor.layers import dsa_trace
from vllm.model_executor.layers import sparse_attn_indexer as sparse_indexer
from vllm.platforms import current_platform
from vllm.utils.import_utils import has_deep_gemm
from vllm.utils.math_utils import cdiv
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.attention.backends.mla.indexer import (
    DeepseekV32IndexerMetadataBuilder,
    get_max_prefill_buffer_size,
)
from vllm.v1.kv_cache_interface import MLAAttentionSpec
from vllm.v1.worker.workspace import init_workspace_manager, reset_workspace_manager

pytestmark = [
    pytest.mark.skipif(not current_platform.is_cuda(), reason="CUDA only"),
    pytest.mark.skipif(not has_deep_gemm(), reason="DeepGEMM not available"),
    pytest.mark.skipif(
        not (current_platform.is_cuda() and current_platform.has_device_capability(90)),
        reason="DeepGEMM MQA logits need Hopper or Blackwell",
    ),
]

# --------------------------------------------------------------------------- #
# Configuration (DeepSeek-V3.2 indexer geometry)
# --------------------------------------------------------------------------- #

HEAD_DIM = 128
N_HEADS = 64
K = 2048
BLOCK = 64  # indexer KV block size (get_supported_kernel_block_sizes)
QUANT_BLOCK = 128
CACHE_ROW = HEAD_DIM + HEAD_DIM // QUANT_BLOCK * 4  # 132: fp8 key + fp32 scale
SCALE_FMT = "ue8m0"
MAX_MODEL_LEN = 163840  # DeepSeek-V3.2
MAX_TOKENS = 8192
MAX_SEQS = 128
BLOCK_TABLE_WIDTH = cdiv(MAX_MODEL_LEN, BLOCK)
CONTEXT_LENS = (2047, 2048, 2049, 4096)
LAYERS = (0, 1)
GATE_LOGITS_MB = "8"  # forces every gate prefill into >= 2 indexer chunks
ROW_BYTES = 8 * K + 16  # ids + scores + tau + valid_count + violations + prefix_len

GATE_OUT = os.environ.get("TOLLBOOTH_GATE_OUT")
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
READ_TRACE = os.path.join(REPO_ROOT, "tools", "read_trace.py")


def layer_name(layer: int) -> str:
    return f"model.layers.{layer}.self_attn.indexer.k_cache"


def mla_layer_name(layer: int) -> str:
    return f"model.layers.{layer}.self_attn.attn"


def record_fact(name: str, data: dict) -> None:
    """Append a fact block to gate-results.json (used to write REPORT.md)."""
    if not GATE_OUT:
        return
    os.makedirs(GATE_OUT, exist_ok=True)
    path = os.path.join(GATE_OUT, "gate-results.json")
    facts = {}
    if os.path.exists(path):
        with open(path) as f:
            facts = json.load(f)
    facts[name] = data
    with open(path, "w") as f:
        json.dump(facts, f, indent=2, sort_keys=True)


def trace_dir(tmp_path_factory, name: str) -> str:
    if GATE_OUT:
        d = os.path.join(GATE_OUT, "traces", name)
        shutil.rmtree(d, ignore_errors=True)
        os.makedirs(d)
        return d
    return str(tmp_path_factory.mktemp(name))


# --------------------------------------------------------------------------- #
# Environment: workspace manager, config, metadata builder
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def env():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    init_workspace_manager(device)
    cfg = VllmConfig()
    # The indexer metadata builder reads exactly one model_config field.
    cfg.model_config = SimpleNamespace(max_model_len=MAX_MODEL_LEN)
    cfg.scheduler_config.max_num_batched_tokens = MAX_TOKENS
    cfg.scheduler_config.max_num_seqs = MAX_SEQS
    spec = MLAAttentionSpec(
        block_size=BLOCK, num_kv_heads=1, head_size=CACHE_ROW, dtype=torch.uint8
    )
    builder = DeepseekV32IndexerMetadataBuilder(
        spec,
        [layer_name(i) for i in LAYERS],
        cfg,
        device,
        block_table_width=BLOCK_TABLE_WIDTH,
    )
    yield SimpleNamespace(
        device=device,
        cfg=cfg,
        spec=spec,
        builder=builder,
        max_total_seq_len=get_max_prefill_buffer_size(cfg),
    )
    reset_workspace_manager()


# --------------------------------------------------------------------------- #
# Synthetic batch: requests, KV caches, per-layer q / k / weights
# --------------------------------------------------------------------------- #


@dataclass
class LayerInputs:
    k: torch.Tensor  # bf16 [T, HEAD_DIM]: keys of this forward's tokens
    q_fp8: torch.Tensor  # fp8 [T, N_HEADS, HEAD_DIM]
    weights: torch.Tensor  # f32 [T, N_HEADS], signed
    # uint8 [num_blocks, BLOCK, CACHE_ROW], contexts inserted
    kv_cache_base: torch.Tensor


@dataclass
class Batch:
    req_ids: list[str]
    ctx_lens: list[int]  # full context length per request (keys scored)
    num_computed: list[int]
    num_scheduled: list[int]
    token_ids: np.ndarray
    rows: list[dsa_trace.RowMeta]
    common: CommonAttentionMetadata
    metadata: object  # DeepseekV32IndexerMetadata
    hidden: torch.Tensor
    layers: dict[int, LayerInputs] = field(default_factory=dict)
    num_blocks: int = 0

    @property
    def num_tokens(self) -> int:
        return sum(self.num_scheduled)


def _keys_random(gen: torch.Generator, n: int, device) -> torch.Tensor:
    return torch.randn(n, HEAD_DIM, generator=gen, device=device).to(torch.bfloat16)


def _keys_tie_design(gen: torch.Generator, n: int, device) -> torch.Tensor:
    """Keys collinear with a unit direction u so that score(pos) = a[pos]*|q.u|:
    n_hi distinct high magnitudes, n_tie *bit-identical* mid keys (a = 2.0),
    the rest low. With n_hi < k the top-k boundary falls inside the tie block."""
    u = torch.randn(HEAD_DIM, generator=gen, device=device)
    u = u / u.norm()
    n_hi = min(1500, n // 3)
    n_tie = min(1000, n // 4)
    a = torch.empty(n, device=device)
    a[:n_hi] = 3.0 + torch.rand(n_hi, generator=gen, device=device)
    a[n_hi : n_hi + n_tie] = 2.0
    a[n_hi + n_tie :] = 0.5 + 0.5 * torch.rand(
        n - n_hi - n_tie, generator=gen, device=device
    )
    perm = torch.randperm(n, generator=gen, device=device)
    a = a[perm]
    return (a[:, None] * u[None, :]).to(torch.bfloat16), u


def make_batch(
    env,
    decode_ctx: list[int],
    prefill_ctx: list[int],
    seed: int,
    logits_mb: str = GATE_LOGITS_MB,
    tie: bool = False,
    session: dsa_trace.TraceSession | None = None,
) -> Batch:
    """Decode requests first (batch order after reordering), then prefills.
    Decode request: context L, one query row at position L-1, L-1 keys already
    in the cache. Prefill request: full prefill, L query rows, keys inserted by
    the forward itself."""
    device = env.device
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    ctx_lens = list(decode_ctx) + list(prefill_ctx)
    num_reqs = len(ctx_lens)
    num_computed = [L - 1 for L in decode_ctx] + [0] * len(prefill_ctx)
    num_scheduled = [1] * len(decode_ctx) + list(prefill_ctx)
    req_ids = [f"dec-{seed}-{i}" for i in range(len(decode_ctx))] + [
        f"pre-{seed}-{i}" for i in range(len(prefill_ctx))
    ]
    T = sum(num_scheduled)
    max_len = max(ctx_lens)

    # Block tables: a random permutation of physical blocks, so that the paged
    # decode path and the prefill gather both go through a non-identity map.
    blocks_per_req = [cdiv(L, BLOCK) for L in ctx_lens]
    num_blocks = sum(blocks_per_req) + 8
    perm = torch.randperm(num_blocks, generator=gen, device=device).cpu()
    block_table = torch.zeros(num_reqs, BLOCK_TABLE_WIDTH, dtype=torch.int32)
    off = 0
    for r, nb in enumerate(blocks_per_req):
        block_table[r, :nb] = perm[off : off + nb].to(torch.int32)
        off += nb

    def slots_for(r: int, positions: torch.Tensor) -> torch.Tensor:
        blk = block_table[r, positions // BLOCK].to(torch.int64)
        return blk * BLOCK + (positions % BLOCK)

    slot_rows = []
    for r in range(num_reqs):
        c, s = num_computed[r], num_scheduled[r]
        slot_rows.append(slots_for(r, torch.arange(c, c + s)))
    slot_mapping = torch.cat(slot_rows).to(device)
    qsl_cpu = torch.zeros(num_reqs + 1, dtype=torch.int32)
    qsl_cpu[1:] = torch.cumsum(torch.tensor(num_scheduled, dtype=torch.int32), 0)
    seq_lens_cpu = torch.tensor(ctx_lens, dtype=torch.int32)
    common = CommonAttentionMetadata(
        query_start_loc=qsl_cpu.to(device),
        query_start_loc_cpu=qsl_cpu,
        seq_lens=seq_lens_cpu.to(device),
        num_reqs=num_reqs,
        num_actual_tokens=T,
        max_query_len=max(num_scheduled),
        max_seq_len=max_len,
        block_table_tensor=block_table.to(device),
        slot_mapping=slot_mapping,
        seq_lens_cpu_upper_bound=seq_lens_cpu,
    )
    prev = os.environ.get("VLLM_SPARSE_INDEXER_MAX_LOGITS_MB")
    os.environ["VLLM_SPARSE_INDEXER_MAX_LOGITS_MB"] = logits_mb
    try:
        metadata = env.builder.build(0, common)
    finally:
        if prev is None:
            os.environ.pop("VLLM_SPARSE_INDEXER_MAX_LOGITS_MB", None)
        else:
            os.environ["VLLM_SPARSE_INDEXER_MAX_LOGITS_MB"] = prev

    # Host-side row identity through the real RowBuilder / PrefixLedger.
    token_ids = (
        np.random.default_rng(seed)
        .integers(1, 50000, size=(num_reqs, max_len))
        .astype(np.int32)
    )
    ledger = session.ledger if session is not None else dsa_trace.PrefixLedger()
    keys = session.request_keys if session is not None else {}
    rows = dsa_trace.RowBuilder(ledger, keys).build(
        req_ids, num_scheduled, num_computed, token_ids
    )
    assert len(rows) == T

    batch = Batch(
        req_ids=req_ids,
        ctx_lens=ctx_lens,
        num_computed=num_computed,
        num_scheduled=num_scheduled,
        token_ids=token_ids,
        rows=rows,
        common=common,
        metadata=metadata,
        hidden=torch.empty(T, 1, device=device),
        num_blocks=num_blocks,
    )

    for layer in LAYERS:
        kv_cache = torch.zeros(
            num_blocks, BLOCK, CACHE_ROW, dtype=torch.uint8, device=device
        )
        k_forward = []
        u_dir = None
        for r, L in enumerate(ctx_lens):
            if tie:
                keys_r, u_dir = _keys_tie_design(gen, L, device)
            else:
                keys_r = _keys_random(gen, L, device)
            c = num_computed[r]
            if c > 0:  # context already in the cache before this forward
                ops.indexer_k_quant_and_cache(
                    keys_r[:c].contiguous(),
                    kv_cache,
                    slots_for(r, torch.arange(0, c)).to(device),
                    QUANT_BLOCK,
                    SCALE_FMT,
                )
            k_forward.append(keys_r[c : c + num_scheduled[r]])
        k = torch.cat(k_forward).contiguous()
        if tie:
            q = u_dir[None, None, :].expand(T, N_HEADS, HEAD_DIM).contiguous()
            weights = torch.full((T, N_HEADS), 1.0 / N_HEADS, device=device)
        else:
            q = torch.randn(T, N_HEADS, HEAD_DIM, generator=gen, device=device)
            weights = torch.randn(T, N_HEADS, generator=gen, device=device) * (
                1.0 / N_HEADS
            )
        batch.layers[layer] = LayerInputs(
            k=k,
            q_fp8=q.to(torch.float8_e4m3fn).contiguous(),
            weights=weights.to(torch.float32).contiguous(),
            kv_cache_base=kv_cache,
        )
    return batch


# --------------------------------------------------------------------------- #
# Running one indexer layer through sparse_attn_indexer()
# --------------------------------------------------------------------------- #


@dataclass
class LogitsCall:
    kind: str  # "prefill" | "decode"
    logits: torch.Tensor  # owned clone of the live FP32 logits
    ks: torch.Tensor | None = None
    ke: torch.Tensor | None = None
    seq_lens: torch.Tensor | None = None


class LogitsSpy:
    """Record the live FP32 logits of every DeepGEMM call. Calls the real
    kernels unchanged; the clone happens on the same stream right after."""

    def __init__(self) -> None:
        self.calls: list[LogitsCall] = []

    def __enter__(self):
        self._orig = (
            sparse_indexer.fp8_fp4_mqa_logits,
            sparse_indexer.fp8_fp4_paged_mqa_logits,
        )
        real_prefill, real_decode = self._orig

        def prefill(q, kv, weights, ks, ke, clean_logits):
            out = real_prefill(q, kv, weights, ks, ke, clean_logits=clean_logits)
            self.calls.append(
                LogitsCall("prefill", out.clone(), ks=ks.clone(), ke=ke.clone())
            )
            return out

        def decode(q, kv_cache, weights, seq_lens, *args, **kwargs):
            out = real_decode(q, kv_cache, weights, seq_lens, *args, **kwargs)
            self.calls.append(
                LogitsCall("decode", out.clone(), seq_lens=seq_lens.clone())
            )
            return out

        sparse_indexer.fp8_fp4_mqa_logits = prefill
        sparse_indexer.fp8_fp4_paged_mqa_logits = decode
        return self

    def __exit__(self, *exc):
        sparse_indexer.fp8_fp4_mqa_logits, sparse_indexer.fp8_fp4_paged_mqa_logits = (
            self._orig
        )
        return False


def run_layer(
    env,
    batch: Batch,
    layer: int,
    kv_cache: torch.Tensor,
    buf: torch.Tensor,
    trace: tuple[dsa_trace.TraceSession, dsa_trace.TraceContext] | None,
    dense_bypass: bool = False,
) -> torch.Tensor:
    attn = {layer_name(layer): batch.metadata}
    dense = ""
    if dense_bypass:
        attn[mla_layer_name(layer)] = SimpleNamespace(
            num_decode_tokens=0, prefill=SimpleNamespace(use_dense_mha=True)
        )
        dense = mla_layer_name(layer)
    fwd = ForwardContext(
        no_compile_layers={},
        attn_metadata=attn,
        slot_mapping={},
        cudagraph_runtime_mode=CUDAGraphMode.NONE,
    )
    inp = batch.layers[layer]
    if trace is not None:
        dsa_trace.set_active(*trace)
    try:
        with override_forward_context(fwd):
            out = sparse_indexer.sparse_attn_indexer(
                batch.hidden,
                layer_name(layer),
                kv_cache,
                inp.q_fp8,
                None,
                inp.k,
                inp.weights,
                QUANT_BLOCK,
                SCALE_FMT,
                K,
                HEAD_DIM,
                MAX_MODEL_LEN,
                env.max_total_seq_len,
                buf,
                False,  # skip_k_cache_insert: the forward inserts its own keys
                False,  # use_pcp
                dense,
            )
    finally:
        dsa_trace.clear_active()
    return out


def decode_dispatch(num_rows: int, logits_stride0: int) -> str:
    """Mirror of the dispatch in sparse_attn_indexer (for the report)."""
    if (
        current_platform.is_cuda()
        and K in (512, 1024, 2048)
        and num_rows <= 64
        and logits_stride0 % 4 == 0
        and current_platform.has_device_capability(90)
        and not current_platform.is_device_capability_family(120)
    ):
        return "cooperative_topk"
    if current_platform.is_cuda() and K in (512, 1024, 2048):
        return "persistent_topk"
    return "top_k_per_row_decode"


def sorted_ids(buf: torch.Tensor) -> torch.Tensor:
    return torch.sort(buf, dim=1).values


def selection_facts(
    off1: torch.Tensor,
    off2: torch.Tensor,
    on: torch.Tensor,
    exp: list[Expect],
) -> dict:
    """Assertion 1, stated against what the selector kernels themselves can
    reproduce. The prefill and decode top-k kernels on this GPU emit the
    selected ids of a row with prefix > k in a run-to-run *different order*
    (see test_decode_dispatch_run_to_run_stability and the report), and when
    the k-th score is tied they pick a run-to-run different subset of the tied
    ids. So: bitwise identity hook-off vs hook-on is asserted for every row
    whose order the kernel reproduces (prefix <= k: the kernels' short-row
    path emits ids in position order), set identity for every row whose top-k
    set is unique (no tie at the k-th score), and every row is additionally
    validated in check_records (trace == this forward's buffer verbatim, tau ==
    k-th live score, nothing selected below it). Empirical agreement counts
    between the three forwards are reported for the record."""
    T = off1.shape[0]
    prefix = torch.zeros(T, dtype=torch.int64, device=off1.device)
    unique = torch.zeros(T, dtype=torch.bool, device=off1.device)
    for e in exp:
        prefix[e.row] = e.prefix_len
        unique[e.row] = e.unique_set
    order_repro = prefix <= K
    assert torch.equal(off1[order_repro], on[order_repro]), (
        "hook changed a selection the kernel reproduces bitwise"
    )
    assert torch.equal(off1[order_repro], off2[order_repro])
    assert torch.equal(sorted_ids(off1)[unique], sorted_ids(on)[unique]), (
        "hook changed a uniquely determined top-k set"
    )
    assert torch.equal(sorted_ids(off1)[unique], sorted_ids(off2)[unique])
    order_12 = (off1 == off2).all(dim=1)
    order_1on = (off1 == on).all(dim=1)
    set_12 = (sorted_ids(off1) == sorted_ids(off2)).all(dim=1)
    set_1on = (sorted_ids(off1) == sorted_ids(on)).all(dim=1)
    return {
        "rows": int(T),
        "rows_prefix_le_k_bitwise_asserted": int(order_repro.sum()),
        "rows_unique_topk_set_asserted": int(unique.sum()),
        "rows_prefix_gt_k_order_not_reproducible": int((~order_repro).sum()),
        "rows_tie_at_kth_set_not_reproducible": int((~unique).sum()),
        "empirical_order_agree_off_off": int(order_12.sum()),
        "empirical_order_agree_off_on": int(order_1on.sum()),
        "empirical_set_agree_off_off": int(set_12.sum()),
        "empirical_set_agree_off_on": int(set_1on.sum()),
    }


def window_logits_equal(a: list[LogitsCall], b: list[LogitsCall], md) -> bool:
    """Compare live logits inside each row's causal window only (outside it the
    buffer is uninitialised memory: clean_logits=False)."""
    chunks = md.prefill.chunks if md.prefill is not None else []
    ci = 0
    for x, y in zip(a, b):
        cols = torch.arange(x.logits.shape[1], device=x.logits.device)[None, :]
        if x.kind == "prefill":
            ch = chunks[ci]
            ci += 1
            n = ch.token_end - ch.token_start
            m = (cols >= ch.cu_seqlen_ks[:n, None]) & (cols < ch.cu_seqlen_ke[:n, None])
            if not torch.equal(x.logits[m], y.logits[m]):
                return False
        else:
            n = md.num_decode_tokens
            m = cols < x.seq_lens.reshape(-1)[:n, None]
            if not torch.equal(x.logits[:n][m], y.logits[:n][m]):
                return False
    return True


# --------------------------------------------------------------------------- #
# Expected records from the live buffer + spied logits (plain torch)
# --------------------------------------------------------------------------- #


@dataclass
class Expect:
    meta: dsa_trace.RowMeta
    ids: np.ndarray  # int32 [K]
    scores: np.ndarray  # f32 [K]
    prefix_len: int
    phase: int
    shadow: bool
    kth: float  # k-th largest live score inside the causal window (NaN if < k)
    row: int  # token row in the forward (index into the top-k buffer)
    unique_set: bool  # no tie at the k-th score: the top-k set is unique


def expected_rows(
    batch: Batch,
    buf_snapshot: torch.Tensor,
    calls: list[LogitsCall],
    shadow: bool = False,
    ids_override: list[torch.Tensor] | None = None,
) -> list[Expect]:
    md = batch.metadata
    chunks = md.prefill.chunks if md.prefill is not None else []
    out: list[Expect] = []
    ci = 0
    for idx, call in enumerate(calls):
        if call.kind == "prefill":
            ch = chunks[ci]
            ci += 1
            n = ch.token_end - ch.token_start
            ids = (
                ids_override[idx]
                if ids_override is not None
                else buf_snapshot[ch.token_start : ch.token_end, :K]
            )
            ks = ch.cu_seqlen_ks[:n].to(torch.int64)
            prefix = (ch.cu_seqlen_ke[:n] - ch.cu_seqlen_ks[:n]).to(torch.int64)
            metas = batch.rows[ch.token_start : ch.token_end]
            phase = dsa_trace.Phase.PREFILL
            logits = call.logits
        else:
            n = md.num_decode_tokens
            ids = buf_snapshot[:n, :K]
            ks = torch.zeros(n, dtype=torch.int64, device=ids.device)
            prefix = call.seq_lens.reshape(-1)[:n].to(torch.int64)
            metas = batch.rows[:n]
            phase = dsa_trace.Phase.DECODE
            logits = call.logits[:n]
        assert ids.shape == (n, K)
        ids64 = ids.to(torch.int64)
        col = (ks[:, None] + ids64.clamp(min=0)).clamp(max=logits.shape[1] - 1)
        sc = logits.gather(1, col).to(torch.float32)
        sc = torch.where(ids64 >= 0, sc, torch.full_like(sc, float("nan")))
        # independent k-th largest score of the causal window (the true tau)
        cols = torch.arange(logits.shape[1], device=logits.device)[None, :]
        window = (cols >= ks[:, None]) & (cols < (ks + prefix)[:, None])
        masked = torch.where(window, logits.float(), torch.full_like(logits, -math.inf))
        kk = min(K, masked.shape[1])
        kth = torch.topk(masked, kk, dim=1).values[:, -1]
        kth = torch.where(prefix >= K, kth, torch.full_like(kth, float("nan")))
        n_ge_kth = (masked >= kth[:, None]).sum(dim=1)
        unique = (prefix < K) | (n_ge_kth == K)
        unique_np = unique.cpu().numpy()
        row0 = ch.token_start if call.kind == "prefill" else 0
        ids_np, sc_np, pre_np = (
            ids.cpu().numpy(),
            sc.cpu().numpy(),
            prefix.cpu().numpy(),
        )
        kth_np = kth.cpu().numpy()
        for r in range(n):
            out.append(
                Expect(
                    metas[r],
                    ids_np[r].copy(),
                    sc_np[r].copy(),
                    int(pre_np[r]),
                    int(phase),
                    shadow,
                    float(kth_np[r]),
                    row0 + r,
                    bool(unique_np[r]),
                )
            )
    assert ci == len(chunks), "every prefill chunk must have produced one logits call"
    return out


def check_records(
    rec: np.ndarray, expects: list[Expect], step_id: int, layer: int
) -> dict:
    """Assertions 2-4 (+ flags) for every record of one (step, layer)."""
    h = rec["header"]
    sel = (h["step_id"] == step_id) & (h["layer_id"] == layer)
    got = {
        (int(h["request_key"][i]), int(h["query_position"][i])): i
        for i in np.nonzero(sel)[0]
    }
    assert len(got) == int(sel.sum()), "duplicate (request, position) records"
    assert len(got) == len(expects), (
        f"{len(got)} records vs {len(expects)} expected rows"
    )
    tau_valid_rows = tau_nan_rows = tie_rows = 0
    for e in expects:
        i = got[(e.meta.request_key, e.meta.query_position)]
        ids, sc = rec["ids"][i], rec["scores"][i]
        # (1)/(2) ids verbatim; scores bitwise; NaN exactly at -1
        assert np.array_equal(ids, e.ids), f"ids differ at record {i}"
        valid = ids >= 0
        assert np.array_equal(
            sc[valid].view(np.uint32), e.scores[valid].view(np.uint32)
        ), f"scores differ bitwise at record {i}"
        assert np.all(np.isnan(sc[~valid])) and not np.any(np.isnan(sc[valid]))
        assert np.all(np.isfinite(sc[valid]))
        # (4) prefix identity and counts
        assert int(h["prefix_length"][i]) == e.prefix_len
        vc = int(valid.sum())
        assert int(h["valid_count"][i]) == vc == min(K, e.prefix_len)
        if vc:
            assert int(ids[valid].max()) < e.prefix_len
            assert len(np.unique(ids[valid])) == vc, "selector returned a duplicate id"
        # (3) tau
        flags = int(h["flags"][i])
        tau = float(h["tau"][i])
        if vc == K:
            assert flags & dsa_trace.Flag.TAU_VALID
            assert tau == float(sc[valid].min())
            assert np.float32(tau).view(np.uint32) == sc[valid].min().view(np.uint32)
            # the selection is a legitimate top-k of the window: tau is the
            # k-th largest live score and nothing selected scores below it
            assert np.float32(tau).view(np.uint32) == np.float32(e.kth).view(np.uint32)
            assert np.all(sc[valid] >= np.float32(e.kth))
            tau_valid_rows += 1
            if int((sc[valid] == np.float32(tau)).sum()) >= 2:
                tie_rows += 1
        else:
            assert not (flags & dsa_trace.Flag.TAU_VALID)
            assert math.isnan(tau)
            tau_nan_rows += 1
        assert bool(flags & dsa_trace.Flag.DECODE) == (
            e.phase == dsa_trace.Phase.DECODE
        )
        assert bool(flags & dsa_trace.Flag.SHADOW) == e.shadow
        assert not (flags & dsa_trace.Flag.VIOLATION)
        assert int(h["query_token_id"][i]) == e.meta.query_token_id
        assert int(h["prefix_node_id"][i]) == e.meta.prefix_node_id
        assert int(h["layer_id"][i]) == layer and int(h["tp_rank"][i]) == 0
    return {
        "records": len(expects),
        "tau_valid_rows": tau_valid_rows,
        "tau_nan_rows": tau_nan_rows,
        "tie_rows": tie_rows,
    }


def check_ledger(rec: np.ndarray, ledger_path: str, batch: Batch) -> None:
    ledger = dsa_trace.PrefixLedger.from_file(ledger_path)
    h = rec["header"]
    by_key = {dsa_trace.request_key(r): i for i, r in enumerate(batch.req_ids)}
    for i in range(len(rec)):
        r = by_key.get(int(h["request_key"][i]))
        if r is None:
            continue
        p = int(h["query_position"][i])
        assert (
            ledger.tokens(int(h["prefix_node_id"][i]))
            == batch.token_ids[r, : p + 1].tolist()
        )


def read_trace_check(dir_: str) -> int:
    proc = subprocess.run(
        [sys.executable, READ_TRACE, dir_, "--check", "--limit", "2"],
        capture_output=True,
        text=True,
    )
    if GATE_OUT:
        with open(
            os.path.join(GATE_OUT, f"read_trace-{os.path.basename(dir_)}.log"), "w"
        ) as f:
            f.write(proc.stdout + proc.stderr)
    return proc.returncode


def new_session(dir_: str, device, **kw) -> dsa_trace.TraceSession:
    kw.setdefault("ring_slots", 64)
    kw.setdefault("capacity_rows", MAX_TOKENS)
    return dsa_trace.TraceSession(
        dir_, run_id=1, tp_rank=0, tp_world_size=1, k=K, device=device, **kw
    )


# --------------------------------------------------------------------------- #
# The gate
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("L", CONTEXT_LENS)
def test_gate_mixed_batch(env, tmp_path_factory, L):
    """One decode row (context L) + one full prefill (L rows, split into >= 2
    indexer chunks) per forward, two indexer layers sharing topk_indices_buffer.
    Hook-off and hook-on forwards from identical inputs; assertions 1-4, 7."""
    device = env.device
    tdir = trace_dir(tmp_path_factory, f"gate-L{L}")
    session = new_session(tdir, device)
    batch = make_batch(env, [L], [L], seed=1000 + L, session=session)
    md = batch.metadata
    assert md.num_decodes == 1 and md.num_prefills == 1
    chunks = md.prefill.chunks
    assert len(chunks) >= 2, "prefill must be split into at least two chunks"
    assert not md.decode.requires_padding
    T = batch.num_tokens
    buf = torch.full((T, K), -1, dtype=torch.int32, device=device)
    ctx = dsa_trace.TraceContext(
        run_id=1, step_id=0, tp_rank=0, tp_world_size=1, rows=batch.rows
    )

    off_bufs, off2_bufs, off_logits, on_bufs = {}, {}, {}, {}
    # hook-off forward, both layers, buffer reused across layers
    with LogitsSpy() as spy_off:
        kv_off = {ly: batch.layers[ly].kv_cache_base.clone() for ly in LAYERS}
        for layer in LAYERS:
            n0 = len(spy_off.calls)
            assert dsa_trace.get_active() is None
            run_layer(env, batch, layer, kv_off[layer], buf, None)
            off_bufs[layer] = buf.clone()
            off_logits[layer] = spy_off.calls[n0:]
    # second hook-off forward: run-to-run stability of the selector itself
    kv_off2 = {ly: batch.layers[ly].kv_cache_base.clone() for ly in LAYERS}
    for layer in LAYERS:
        run_layer(env, batch, layer, kv_off2[layer], buf, None)
        off2_bufs[layer] = buf.clone()
    # hook-on forward from identical inputs
    with LogitsSpy() as spy_on:
        kv_on = {ly: batch.layers[ly].kv_cache_base.clone() for ly in LAYERS}
        on_logits = {}
        for layer in LAYERS:
            n0 = len(spy_on.calls)
            run_layer(env, batch, layer, kv_on[layer], buf, (session, ctx))
            on_bufs[layer] = buf.clone()
            on_logits[layer] = spy_on.calls[n0:]
    session.close()
    torch.cuda.synchronize()

    # (1) selection identical hook-off vs hook-on (see selection_facts)
    exp_by_layer = {}
    facts = {
        "L": L,
        "num_tokens": T,
        "chunks": [(c.token_start, c.token_end) for c in chunks],
    }
    for layer in LAYERS:
        exp_by_layer[layer] = expected_rows(batch, on_bufs[layer], on_logits[layer])
        facts[f"selection_layer{layer}"] = selection_facts(
            off_bufs[layer], off2_bufs[layer], on_bufs[layer], exp_by_layer[layer]
        )
        assert torch.equal(kv_off[layer], kv_on[layer])
        # kernel determinism (informative): same logits both runs
        facts[f"window_logits_identical_layer{layer}"] = window_logits_equal(
            off_logits[layer], on_logits[layer], md
        )
        assert facts[f"window_logits_identical_layer{layer}"]
    dec = off_logits[0][-1]
    assert dec.kind == "decode"
    facts["decode_dispatch"] = decode_dispatch(
        md.num_decode_tokens, dec.logits.stride(0)
    )

    # (2)-(4) every record against the live buffer and live logits
    rec = dsa_trace.read_records(session.records_path, K)
    assert len(rec) == len(LAYERS) * T
    for layer in LAYERS:
        exp = exp_by_layer[layer]
        facts[f"layer{layer}"] = check_records(rec, exp, step_id=0, layer=layer)
    # records land in capture order: prefill chunk rows, then the decode row
    h = rec["header"]
    assert h["layer_id"].tolist() == sorted(h["layer_id"].tolist())
    for layer in LAYERS:
        sel = h["layer_id"] == layer
        assert h["query_position"][sel][-1] == L - 1 and (
            h["flags"][sel][-1] & dsa_trace.Flag.DECODE
        )
    decode_rec = rec[(h["flags"] & dsa_trace.Flag.DECODE) != 0]
    facts["decode_tau_valid"] = [
        bool(f & dsa_trace.Flag.TAU_VALID) for f in decode_rec["header"]["flags"]
    ]
    assert all(facts["decode_tau_valid"]) == (L >= K)
    check_ledger(rec, session.ledger_path, batch)
    with open(session.manifest_path) as f:
        manifest = json.load(f)
    assert manifest["records_written"] == len(rec) and manifest["fatal"] is None
    facts["ring_stalls"] = manifest["ring_stalls"]
    # (7) offline reader agrees
    facts["read_trace_check_rc"] = read_trace_check(tdir)
    assert facts["read_trace_check_rc"] == 0
    record_fact(f"gate_L{L}", facts)


def test_gate_ties_keep_selector_order(env, tmp_path_factory):
    """Assertion 3, tie case: bit-identical keys (and fp8 quantisation, which
    collapses nearby magnitudes onto the same codes) so that the top-k boundary
    falls inside a block of equal scores. Hook-on records must carry the
    selector's ids verbatim, in the selector's order, with tau equal to the
    tied k-th score. The kernels pick a different subset of the tied ids run
    to run, which is recorded, not blamed on the hook."""
    device = env.device
    L = 4096
    tdir = trace_dir(tmp_path_factory, "gate-ties")
    session = new_session(tdir, device)
    batch = make_batch(env, [L], [L], seed=77, tie=True, session=session)
    T = batch.num_tokens
    buf = torch.full((T, K), -1, dtype=torch.int32, device=device)
    ctx = dsa_trace.TraceContext(
        run_id=1, step_id=0, tp_rank=0, tp_world_size=1, rows=batch.rows
    )
    facts = {}
    with LogitsSpy() as spy:
        kv = {ly: batch.layers[ly].kv_cache_base.clone() for ly in LAYERS}
        off, calls_off = {}, {}
        for layer in LAYERS:
            n0 = len(spy.calls)
            run_layer(env, batch, layer, kv[layer], buf, None)
            off[layer] = buf.clone()
            calls_off[layer] = spy.calls[n0:]
        # kernel determinism under ties: a second hook-off run
        kv2 = {ly: batch.layers[ly].kv_cache_base.clone() for ly in LAYERS}
        off2 = {}
        for layer in LAYERS:
            run_layer(env, batch, layer, kv2[layer], buf, None)
            off2[layer] = buf.clone()
        spy.calls.clear()
        kv3 = {ly: batch.layers[ly].kv_cache_base.clone() for ly in LAYERS}
        on, calls = {}, {}
        for layer in LAYERS:
            n0 = len(spy.calls)
            run_layer(env, batch, layer, kv3[layer], buf, (session, ctx))
            on[layer] = buf.clone()
            calls[layer] = spy.calls[n0:]
    session.close()
    torch.cuda.synchronize()
    rec = dsa_trace.read_records(session.records_path, K)
    for layer in LAYERS:
        exp = expected_rows(batch, on[layer], calls[layer])
        facts[f"selection_layer{layer}"] = selection_facts(
            off[layer], off2[layer], on[layer], exp
        )
        facts[f"window_logits_identical_layer{layer}"] = window_logits_equal(
            calls_off[layer], calls[layer], batch.metadata
        )
        assert facts[f"window_logits_identical_layer{layer}"]
        f = check_records(rec, exp, 0, layer)
        assert f["tie_rows"] > 0, (
            "tie construction did not produce a tie at the boundary"
        )
        facts[f"layer{layer}"] = f
    # the decode row itself must sit on a tie: several selected scores == tau
    h = rec["header"]
    dsel = (h["flags"] & dsa_trace.Flag.DECODE) != 0
    for i in np.nonzero(dsel)[0]:
        sc = rec["scores"][i]
        valid = rec["ids"][i] >= 0
        n_tied = int((sc[valid] == np.float32(h["tau"][i])).sum())
        assert n_tied >= 2, "decode row has no tie at the boundary"
        facts.setdefault("decode_tied_at_tau", []).append(n_tied)
    assert read_trace_check(tdir) == 0
    record_fact("gate_ties", facts)


def test_gate_ring_lifecycle_under_backpressure(env, tmp_path_factory):
    """Assertion 5: pinned slots, copy stream, CUDA events; a throttled writer
    fills the ring (stalls > 0), capacity_rows smaller than a chunk exercises
    the split path; nothing is dropped or corrupted (checked against an
    in-memory gather of the same live tensors)."""
    device = env.device
    L = 4096
    tdir = trace_dir(tmp_path_factory, "gate-ring")
    session = new_session(tdir, device, ring_slots=2, capacity_rows=200)
    ring = session.ring
    assert isinstance(ring.copy_stream, torch.cuda.Stream)
    for slot in ring.slots:
        for t in (
            slot.h_ids,
            slot.h_scores,
            slot.h_tau,
            slot.h_valid_count,
            slot.h_violations,
            slot.h_prefix_len,
        ):
            assert t.is_pinned()
        assert isinstance(slot.producer_event, torch.cuda.Event)
        assert isinstance(slot.copy_event, torch.cuda.Event)
    real_write = session.records_fd.write
    writes = []

    def slow_write(b):
        time.sleep(0.02)
        writes.append(len(b))
        return real_write(b)

    session.records_fd.write = slow_write  # type: ignore[method-assign]

    batch = make_batch(env, [L], [L], seed=5, session=session)
    chunks = batch.metadata.prefill.chunks
    assert min(c.token_end - c.token_start for c in chunks) > ring.capacity_rows
    T = batch.num_tokens
    buf = torch.full((T, K), -1, dtype=torch.int32, device=device)
    ctx = dsa_trace.TraceContext(
        run_id=1, step_id=3, tp_rank=0, tp_world_size=1, rows=batch.rows
    )
    with LogitsSpy() as spy:
        kv = {ly: batch.layers[ly].kv_cache_base.clone() for ly in LAYERS}
        snaps, calls, refs = {}, {}, {}
        t0 = time.perf_counter()
        for layer in LAYERS:
            n0 = len(spy.calls)
            run_layer(env, batch, layer, kv[layer], buf, (session, ctx))
            snaps[layer] = buf.clone()
            calls[layer] = spy.calls[n0:]
            # in-memory copy of what the hook gathered, from the live tensors
            g = []
            for call, ch in zip(calls[layer][:-1], chunks):
                n = ch.token_end - ch.token_start
                g.append(
                    dsa_trace.gather_scores(
                        call.logits,
                        snaps[layer][ch.token_start : ch.token_end, :K],
                        ch.cu_seqlen_ks[:n],
                        ch.cu_seqlen_ke[:n] - ch.cu_seqlen_ks[:n],
                        K,
                    )
                )
            dcall = calls[layer][-1]
            n = batch.metadata.num_decode_tokens
            g.append(
                dsa_trace.gather_scores(
                    dcall.logits[:n],
                    snaps[layer][:n, :K],
                    torch.zeros(n, dtype=torch.int32, device=device),
                    dcall.seq_lens.reshape(-1)[:n],
                    K,
                )
            )
            refs[layer] = g
        produce_s = time.perf_counter() - t0
    session.close()
    torch.cuda.synchronize()
    assert ring.stalls > 0, "the throttled writer never filled the ring"
    rec = dsa_trace.read_records(session.records_path, K)
    assert len(rec) == len(LAYERS) * T
    h = rec["header"]
    for layer in LAYERS:
        sel = np.nonzero(h["layer_id"] == layer)[0]
        assert len(sel) == T
        ids = np.concatenate([g.ids.cpu().numpy() for g in refs[layer]])
        scores = np.concatenate([g.scores.cpu().numpy() for g in refs[layer]])
        tau = np.concatenate([g.tau.cpu().numpy() for g in refs[layer]])
        vc = np.concatenate([g.valid_count.cpu().numpy() for g in refs[layer]])
        pl = np.concatenate([g.prefix_len.cpu().numpy() for g in refs[layer]])
        assert np.array_equal(rec["ids"][sel], ids)
        assert np.array_equal(
            rec["scores"][sel].view(np.uint32), scores.view(np.uint32)
        )
        assert np.array_equal(h["tau"][sel].view(np.uint32), tau.view(np.uint32))
        assert np.array_equal(h["valid_count"][sel], vc) and np.array_equal(
            h["prefix_length"][sel], pl
        )
        n_dec = batch.metadata.num_decode_tokens
        capture_order = batch.rows[n_dec:] + batch.rows[:n_dec]
        assert np.array_equal(
            h["query_position"][sel],
            np.array([m.query_position for m in capture_order]),
        )
        assert not np.any(h["flags"][sel] & dsa_trace.Flag.VIOLATION)
        # the plain-torch expectation too (independent of gather_scores)
        check_records(rec, expected_rows(batch, snaps[layer], calls[layer]), 3, layer)
    with open(session.manifest_path) as f:
        manifest = json.load(f)
    assert manifest["ring_stalls"] == ring.stalls and manifest[
        "records_written"
    ] == len(rec)
    assert read_trace_check(tdir) == 0
    record_fact(
        "gate_ring",
        {
            "ring_slots": 2,
            "capacity_rows": 200,
            "chunk_rows": [c.token_end - c.token_start for c in chunks],
            "stalls": ring.stalls,
            "slot_writes": len(writes),
            "records": len(rec),
            "producer_wall_s": produce_s,
        },
    )


def test_gate_dense_bypass_shadow(env, tmp_path_factory):
    """Assertion 6: with the main MLA on the dense short-prefill path, the live
    buffer stays untouched, records are flagged shadow_dense, and the shadow
    ids equal the selector's output on the same logits (live path, same inputs,
    and a direct top_k_per_row_prefill over the shadow logits)."""
    device = env.device
    L = 2049
    tdir = trace_dir(tmp_path_factory, "gate-shadow")
    session = new_session(tdir, device)
    batch = make_batch(env, [], [L], seed=9, session=session)
    assert batch.metadata.num_decodes == 0 and len(batch.metadata.prefill.chunks) >= 2
    T = batch.num_tokens
    ctx = dsa_trace.TraceContext(
        run_id=1, step_id=0, tp_rank=0, tp_world_size=1, rows=batch.rows
    )
    live = torch.full((T, K), 17, dtype=torch.int32, device=device)
    with LogitsSpy() as spy:
        kv = {ly: batch.layers[ly].kv_cache_base.clone() for ly in LAYERS}
        shadow_calls = {}
        for layer in LAYERS:
            n0 = len(spy.calls)
            out = run_layer(
                env, batch, layer, kv[layer], live, (session, ctx), dense_bypass=True
            )
            assert out is live
            shadow_calls[layer] = spy.calls[n0:]
        assert torch.all(live == 17), "shadow scoring touched the live top-k buffer"
        # reference: the live path on identical inputs
        buf = torch.full((T, K), -1, dtype=torch.int32, device=device)
        kv2 = {ly: batch.layers[ly].kv_cache_base.clone() for ly in LAYERS}
        live_sel, live_calls = {}, {}
        for layer in LAYERS:
            n0 = len(spy.calls)
            run_layer(env, batch, layer, kv2[layer], buf, None)
            live_sel[layer] = buf.clone()
            live_calls[layer] = spy.calls[n0:]
    session.close()
    torch.cuda.synchronize()
    rec = dsa_trace.read_records(session.records_path, K)
    assert len(rec) == len(LAYERS) * T
    assert np.all(rec["header"]["flags"] & dsa_trace.Flag.SHADOW)
    facts = {"L": L}
    h = rec["header"]
    for layer in LAYERS:
        assert window_logits_equal(
            shadow_calls[layer], live_calls[layer], batch.metadata
        )
        # the records' own ids, in chunk order, as the reference selection:
        # scores / tau / k-th score / flags are then validated against the
        # shadow logits by check_records
        rec_ids = torch.from_numpy(rec["ids"][h["layer_id"] == layer].copy()).to(device)
        per_chunk = [
            rec_ids[ch.token_start : ch.token_end]
            for ch in batch.metadata.prefill.chunks
        ]
        exp = expected_rows(
            batch, rec_ids, shadow_calls[layer], shadow=True, ids_override=per_chunk
        )
        facts[f"layer{layer}"] = check_records(rec, exp, 0, layer)
        # top_k_per_row_prefill run directly on the shadow logits
        direct = []
        for call, ch in zip(shadow_calls[layer], batch.metadata.prefill.chunks):
            n = ch.token_end - ch.token_start
            outk = torch.full((n, K), -1, dtype=torch.int32, device=device)
            ops.top_k_per_row_prefill(
                call.logits,
                ch.cu_seqlen_ks,
                ch.cu_seqlen_ke,
                outk,
                n,
                call.logits.stride(0),
                call.logits.stride(1),
                K,
            )
            direct.append(outk)
        direct = torch.cat(direct)
        # shadow ids == live selector ids on the same inputs == direct rerun:
        # bitwise where the kernel order is reproducible, as sets everywhere
        prefix = torch.tensor([e.prefix_len for e in exp], device=device)
        repro = prefix <= K
        assert torch.equal(rec_ids[repro], live_sel[layer][repro])
        assert torch.equal(rec_ids[repro], direct[repro])
        assert torch.equal(sorted_ids(rec_ids), sorted_ids(live_sel[layer]))
        assert torch.equal(sorted_ids(rec_ids), sorted_ids(direct))
        facts[f"layer{layer}"]["rows_bitwise_vs_live"] = int(
            (rec_ids == live_sel[layer]).all(dim=1).sum()
        )
    assert read_trace_check(tdir) == 0
    record_fact("gate_shadow", facts)


def test_decode_dispatch_run_to_run_stability(env, tmp_path_factory):
    """Characterise the decode top-k kernels sparse_attn_indexer dispatches to
    on this GPU (no hook involved): is the selected set stable across three
    identical forwards, and is the emitted order stable? Recorded for the
    report; the set must be stable."""
    device = env.device
    facts = {}
    for n_rows in (64, 96):  # <= 64 rows: cooperative_topk; > 64: persistent_topk
        batch = make_batch(env, [4096] * n_rows, [], seed=500 + n_rows, logits_mb="512")
        buf = torch.full((n_rows, K), -1, dtype=torch.int32, device=device)
        runs = []
        for _ in range(3):
            kv = batch.layers[0].kv_cache_base.clone()
            with LogitsSpy() as spy:
                run_layer(env, batch, 0, kv, buf, None)
            runs.append(buf.clone())
        stride0 = spy.calls[-1].logits.stride(0)
        dispatch = decode_dispatch(n_rows, stride0)
        set_stable = all(
            torch.equal(sorted_ids(runs[0]), sorted_ids(r)) for r in runs[1:]
        )
        order_stable_rows = int(
            ((runs[0] == runs[1]) & (runs[0] == runs[2])).all(dim=1).sum()
        )
        facts[dispatch] = {
            "rows": n_rows,
            "L": 4096,
            "set_stable_3_runs": set_stable,
            "order_stable_rows_3_runs": order_stable_rows,
        }
        assert set_stable, f"{dispatch}: selected set differs run to run"
        del batch, kv
        torch.cuda.empty_cache()
    record_fact("decode_dispatch_stability", facts)


# --------------------------------------------------------------------------- #
# Overhead numbers (POD_BRIEF section 4)
# --------------------------------------------------------------------------- #


def _time_capture(session: dsa_trace.TraceSession, fn, iters: int = 20) -> dict:
    """CUDA-event time on the producer stream around the capture call (gather +
    copy enqueue), CUDA-event time until the D2H copy on the copy stream has
    landed, and host wall time of the call. Medians over ``iters``."""
    gather_ms, total_ms, host_us = [], [], []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = torch.cuda.Event(enable_timing=True)
        t1 = torch.cuda.Event(enable_timing=True)
        t2 = torch.cuda.Event(enable_timing=True)
        t0.record()
        h0 = time.perf_counter()
        fn()
        h1 = time.perf_counter()
        t1.record()
        t2.record(session.ring.copy_stream)
        torch.cuda.synchronize()
        gather_ms.append(t0.elapsed_time(t1))
        total_ms.append(t0.elapsed_time(t2))
        host_us.append((h1 - h0) * 1e6)
    session.flush()
    return {
        "iters": iters,
        "gather_enqueue_us_median": float(np.median(gather_ms) * 1e3),
        "gather_enqueue_us_min": float(np.min(gather_ms) * 1e3),
        "until_copy_landed_us_median": float(np.median(total_ms) * 1e3),
        "host_call_us_median": float(np.median(host_us)),
    }


@pytest.mark.parametrize("L", (4096, MAX_MODEL_LEN))
def test_overhead_decode_64_rows(env, tmp_path_factory, L):
    device = env.device
    tdir = trace_dir(tmp_path_factory, f"overhead-decode-L{L}")
    session = new_session(tdir, device)
    n = 64
    batch = make_batch(
        env, [L] * n, [], seed=300 + L % 1000, logits_mb="512", session=session
    )
    md = batch.metadata
    assert md.num_decode_tokens == n and md.num_prefills == 0
    buf = torch.full((n, K), -1, dtype=torch.int32, device=device)
    ctx = dsa_trace.TraceContext(
        run_id=1, step_id=0, tp_rank=0, tp_world_size=1, rows=batch.rows
    )
    with LogitsSpy() as spy:
        kv = batch.layers[0].kv_cache_base.clone()
        run_layer(env, batch, 0, kv, buf, None)
    logits = spy.calls[-1].logits
    trace = (session, ctx)
    res = _time_capture(
        session,
        lambda: dsa_trace.capture_decode(
            trace, 0, logits, buf, md.decode.seq_lens, n, md.decode.requires_padding
        ),
    )
    session.close()
    res.update(
        {
            "rows": n,
            "L": L,
            "bytes_moved_per_capture": n * ROW_BYTES,
            "record_bytes_per_capture": n * dsa_trace.record_size(K),
            "decode_dispatch": decode_dispatch(n, logits.stride(0)),
            "logits_shape": list(logits.shape),
        }
    )
    record_fact(f"overhead_decode64_L{L}", res)
    del batch, kv
    torch.cuda.empty_cache()


def test_overhead_prefill_4096_chunk(env, tmp_path_factory):
    device = env.device
    tdir = trace_dir(tmp_path_factory, "overhead-prefill4096")
    session = new_session(tdir, device)
    L = 4096
    batch = make_batch(env, [], [L], seed=44, logits_mb="512", session=session)
    chunks = batch.metadata.prefill.chunks
    assert len(chunks) == 1 and chunks[0].token_end - chunks[0].token_start == L
    buf = torch.full((L, K), -1, dtype=torch.int32, device=device)
    ctx = dsa_trace.TraceContext(
        run_id=1, step_id=0, tp_rank=0, tp_world_size=1, rows=batch.rows
    )
    with LogitsSpy() as spy:
        kv = batch.layers[0].kv_cache_base.clone()
        run_layer(env, batch, 0, kv, buf, None)
    logits = spy.calls[0].logits
    trace = (session, ctx)
    res = _time_capture(
        session,
        lambda: dsa_trace.capture_prefill_chunk(trace, 0, logits, buf, chunks[0]),
    )
    session.close()
    res.update(
        {
            "rows": L,
            "bytes_moved_per_capture": L * ROW_BYTES,
            "record_bytes_per_capture": L * dsa_trace.record_size(K),
            "logits_shape": list(logits.shape),
        }
    )
    record_fact("overhead_prefill4096", res)


def test_overhead_writer_throughput(env, tmp_path_factory):
    """Sustained records-file MB/s of the writer thread on this box's local
    disk, driven by back-to-back 4096-row captures (ring never stalls with 64
    slots x 8192 rows unless the writer is the bottleneck)."""
    device = env.device
    tdir = trace_dir(tmp_path_factory, "overhead-writer")
    session = new_session(tdir, device)
    L = 4096
    batch = make_batch(env, [], [L], seed=45, logits_mb="512", session=session)
    chunk = batch.metadata.prefill.chunks[0]
    buf = torch.full((L, K), -1, dtype=torch.int32, device=device)
    ctx = dsa_trace.TraceContext(
        run_id=1, step_id=0, tp_rank=0, tp_world_size=1, rows=batch.rows
    )
    with LogitsSpy() as spy:
        kv = batch.layers[0].kv_cache_base.clone()
        run_layer(env, batch, 0, kv, buf, None)
    logits = spy.calls[0].logits
    captures = 40
    t0 = time.perf_counter()
    for i in range(captures):
        dsa_trace.capture_prefill_chunk((session, ctx), i, logits, buf, chunk)
    t_enqueued = time.perf_counter()
    session.flush()
    t_flushed = time.perf_counter()
    session.close()
    t_closed = time.perf_counter()
    nbytes = session.writer.bytes_written
    assert nbytes == captures * L * dsa_trace.record_size(K)
    st = os.statvfs(tdir)
    res = {
        "captures": captures,
        "rows_per_capture": L,
        "bytes_written": nbytes,
        "enqueue_s": t_enqueued - t0,
        "flush_s": t_flushed - t0,
        "close_fsync_s": t_closed - t0,
        "MB_per_s_to_flush": nbytes / (t_flushed - t0) / 1e6,
        "MB_per_s_to_fsync": nbytes / (t_closed - t0) / 1e6,
        "ring_stalls": session.ring.stalls,
        "trace_dir": tdir,
        "fs_block_size": st.f_bsize,
    }
    record_fact("overhead_writer", res)
    os.remove(session.records_path)  # 2.7 GB; the manifest stays
