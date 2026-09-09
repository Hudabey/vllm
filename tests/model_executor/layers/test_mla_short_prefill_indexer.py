# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math
from types import SimpleNamespace

import pytest
import torch

import vllm.model_executor.layers.sparse_attn_indexer as sparse_indexer
from vllm.config import CUDAGraphMode
from vllm.models.deepseek_v32 import attention as deepseek_v32_attention
from vllm.models.deepseek_v32.attention import DeepseekV32Attention
from vllm.v1.attention.backends.mla.indexer import DeepseekV32IndexerMetadata

INDEXER_LAYER = "model.layers.0.self_attn.indexer.k_cache"
MLA_LAYER = "model.layers.0.self_attn.attn"


def make_indexer_metadata(
    *,
    num_decodes: int = 0,
    num_decode_tokens: int = 0,
    num_prefills: int = 1,
    num_prefill_tokens: int = 1,
    slot_mapping: torch.Tensor | None = None,
) -> DeepseekV32IndexerMetadata:
    if slot_mapping is None:
        slot_mapping = torch.zeros(num_prefill_tokens, dtype=torch.long)
    return DeepseekV32IndexerMetadata(
        seq_lens=torch.empty(0, dtype=torch.int32),
        max_seq_len=2048,
        slot_mapping=slot_mapping,
        num_decodes=num_decodes,
        num_decode_tokens=num_decode_tokens,
        num_prefills=num_prefills,
        num_prefill_tokens=num_prefill_tokens,
        prefill=SimpleNamespace(chunks=[]) if num_prefills else None,
    )


def make_mla_metadata(*, use_dense_mha: bool = True, num_decode_tokens: int = 0):
    return SimpleNamespace(
        num_decode_tokens=num_decode_tokens,
        prefill=SimpleNamespace(use_dense_mha=use_dense_mha),
    )


@pytest.mark.parametrize(
    "batch_kind",
    [
        "short",
        "threshold_mismatch",
        "force_mqa",
        "mla_decode",
        "capture",
        "full",
    ],
)
def test_short_prefill_updates_k_cache_before_scoring_decision(
    monkeypatch: pytest.MonkeyPatch,
    batch_kind: str,
):
    slot_mapping = torch.tensor([63, 64, 127, 128, -1])
    mla_num_decode_tokens = 1 if batch_kind == "mla_decode" else 0
    runtime_mode = (
        CUDAGraphMode.FULL if batch_kind == "full" else CUDAGraphMode.PIECEWISE
    )
    should_skip = batch_kind in ("short", "threshold_mismatch")
    num_decodes = int(batch_kind == "threshold_mismatch")
    num_decode_tokens = 3 if batch_kind == "threshold_mismatch" else 0
    num_prefills = 0 if batch_kind == "threshold_mismatch" else 2
    num_prefill_tokens = 0 if batch_kind == "threshold_mismatch" else 5
    if batch_kind == "threshold_mismatch":
        # With MTP=3 the indexer threshold is four. A main MLA backend whose
        # threshold is one (for example FlashMLA under DCP) still routes this
        # three-token extend through dense prefill attention.
        slot_mapping = slot_mapping[:3]
    indexer_metadata = make_indexer_metadata(
        num_decodes=num_decodes,
        num_decode_tokens=num_decode_tokens,
        num_prefills=num_prefills,
        num_prefill_tokens=num_prefill_tokens,
        slot_mapping=slot_mapping,
    )
    if indexer_metadata.num_decodes:
        indexer_metadata.decode = object()
    mla_metadata = make_mla_metadata(
        use_dense_mha=batch_kind != "force_mqa",
        num_decode_tokens=mla_num_decode_tokens,
    )

    observed: dict[str, object] = {}

    monkeypatch.setattr(
        sparse_indexer,
        "get_forward_context",
        lambda: SimpleNamespace(
            attn_metadata={
                INDEXER_LAYER: indexer_metadata,
                MLA_LAYER: mla_metadata,
            },
            cudagraph_runtime_mode=runtime_mode,
        ),
    )
    monkeypatch.setattr(
        sparse_indexer.current_platform, "fp8_dtype", lambda: torch.float16
    )
    monkeypatch.setattr(
        torch.cuda,
        "is_current_stream_capturing",
        lambda: batch_kind == "capture",
    )

    def record_cache_update(k, kv_cache, slots, block_size, scale_fmt):
        observed.update(k=k.clone(), slots=slots)

    monkeypatch.setattr(
        sparse_indexer.ops, "indexer_k_quant_and_cache", record_cache_update
    )

    class ScoringReached(Exception):
        pass

    def scoring_trigger():
        if should_skip:
            pytest.fail("short dense-MHA prefill must not enter indexer scoring")
        raise ScoringReached

    def scoring_decode(*args):
        raise ScoringReached

    monkeypatch.setattr(sparse_indexer, "current_workspace_manager", scoring_trigger)
    monkeypatch.setattr(
        sparse_indexer,
        "kv_cache_as_quant_view",
        scoring_decode,
    )

    hidden_states = torch.full((7, 1), float("inf"))
    k = torch.arange(28, dtype=torch.float32).reshape(7, 4)
    topk_indices = torch.full((7, 2048), 17, dtype=torch.int32)

    def run_indexer():
        assert DeepseekV32Attention.supports_dense_mha_prefill
        return sparse_indexer.sparse_attn_indexer(
            hidden_states,
            INDEXER_LAYER,
            torch.empty(1),
            torch.full((7, 1), float("inf")),
            None,
            k,
            torch.full((7, 1), float("inf")),
            128,
            "ue8m0",
            2048,
            4,
            4096,
            4096,
            topk_indices,
            False,
            False,
            MLA_LAYER,
        )

    if should_skip:
        assert run_indexer() is topk_indices
        assert torch.all(topk_indices == 17)
    else:
        with pytest.raises(ScoringReached):
            run_indexer()
        assert torch.all(topk_indices == -1)

    # K cache is always updated before the scoring decision.
    torch.testing.assert_close(observed["k"], k[: slot_mapping.numel()])
    assert observed["slots"] is slot_mapping


def test_skipped_k_cache_insert_accepts_no_k(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    indexer_metadata = make_indexer_metadata(
        num_prefills=0,
        num_prefill_tokens=0,
        slot_mapping=torch.empty(0, dtype=torch.long),
    )
    monkeypatch.setattr(
        sparse_indexer,
        "get_forward_context",
        lambda: SimpleNamespace(
            attn_metadata={INDEXER_LAYER: indexer_metadata},
            cudagraph_runtime_mode=CUDAGraphMode.PIECEWISE,
        ),
    )
    monkeypatch.setattr(
        sparse_indexer.current_platform, "fp8_dtype", lambda: torch.float16
    )

    topk_indices = torch.full((1, 2048), 17, dtype=torch.int32)
    result = sparse_indexer.sparse_attn_indexer(
        torch.empty(1, 1),
        INDEXER_LAYER,
        torch.empty(1),
        torch.empty(1, 1),
        None,
        None,
        torch.empty(1, 1),
        128,
        "ue8m0",
        2048,
        4,
        4096,
        4096,
        topk_indices,
        True,
        False,
        "",
    )

    assert result is topk_indices
    assert torch.all(topk_indices == -1)


@pytest.mark.parametrize("fp8_query", [False, True])
def test_deepseek_v32_dispatches_selected_mha(
    monkeypatch: pytest.MonkeyPatch,
    fp8_query: bool,
) -> None:
    attn_metadata = SimpleNamespace(num_actual_tokens=2)
    kv_cache = torch.empty(1)
    monkeypatch.setattr(
        deepseek_v32_attention,
        "get_attention_context",
        lambda _: (attn_metadata, None, kv_cache, None),
    )

    observed = {}

    def record_forward_impl(*args):
        observed["args"] = args

    layer = SimpleNamespace(
        indexer=None,
        skip_topk=False,
        layer_name=MLA_LAYER,
        use_pcp=False,
        _fp8_query=fp8_query,
        _use_sparse_mha=lambda _: True,
        rotary_emb=lambda _positions, q: (q + 1, None),
        forward_impl=record_forward_impl,
    )
    q_nope = torch.randn(2, 1, 2)
    q_pe = torch.randn(2, 1, 2)
    mqa_q = torch.randn(2, 1, 2)
    mha_q = torch.cat((q_nope, q_pe + 1 if fp8_query else mqa_q), dim=-1)
    kv_c = torch.empty(2, 2)
    k_pe = torch.empty(2, 2)
    output = torch.empty(2, 2)

    DeepseekV32Attention._sparse_indexer_and_attn(
        layer,
        torch.arange(2),
        torch.empty(2, 2),
        q_nope,
        q_pe,
        None,
        None,
        None,
        kv_c,
        k_pe,
        torch.empty(2, 1, 2),
        mqa_q,
        output,
    )

    expected_args = (
        mha_q,
        kv_c,
        kv_cache,
        attn_metadata,
        output,
    )
    actual_args = observed["args"]
    torch.testing.assert_close(actual_args[0], mha_q)
    assert actual_args[2].shape == (2, 1, 2)
    assert actual_args[2].data_ptr() == k_pe.data_ptr()
    assert all(
        actual is expected
        for actual, expected in zip(
            actual_args[1:2] + actual_args[3:],
            expected_args[1:],
        )
    )


def test_dense_bypass_shadow_scores_into_scratch_and_records(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """tollbooth: with a trace armed, the dense short-prefill bypass must still
    return the live buffer untouched, and the trace must hold shadow-flagged
    records whose ids equal the selector's output on the same logits."""
    from vllm.model_executor.layers import dsa_trace

    rows, cols, k = 3, 6, 4
    chunk = SimpleNamespace(
        token_start=0,
        token_end=rows,
        cu_seqlen_ks=torch.tensor([0, 0, 0], dtype=torch.int32),
        cu_seqlen_ke=torch.tensor([1, 2, 3], dtype=torch.int32),
        local_cu_seq_lens=torch.zeros(2, dtype=torch.int32),
        local_total_seq_lens=3,
        max_local_total_seq_lens=3,
        skip_kv_gather=True,
        block_table=torch.empty(0, dtype=torch.int32),
    )
    indexer_metadata = make_indexer_metadata(
        num_prefills=1,
        num_prefill_tokens=rows,
        slot_mapping=torch.zeros(rows, dtype=torch.long),
    )
    indexer_metadata.prefill = SimpleNamespace(chunks=[chunk])
    mla_metadata = make_mla_metadata(use_dense_mha=True, num_decode_tokens=0)
    monkeypatch.setattr(
        sparse_indexer,
        "get_forward_context",
        lambda: SimpleNamespace(
            attn_metadata={INDEXER_LAYER: indexer_metadata, MLA_LAYER: mla_metadata},
            cudagraph_runtime_mode=CUDAGraphMode.PIECEWISE,
        ),
    )
    monkeypatch.setattr(
        sparse_indexer.current_platform, "fp8_dtype", lambda: torch.float16
    )
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(
        sparse_indexer.ops, "indexer_k_quant_and_cache", lambda *a, **kw: None
    )
    monkeypatch.setattr(
        sparse_indexer, "_gather_workspace_shapes", lambda *a, **kw: (None, None)
    )
    monkeypatch.setattr(
        sparse_indexer,
        "current_workspace_manager",
        lambda: SimpleNamespace(
            get_simultaneous=lambda *specs: tuple(
                torch.empty(0, dtype=torch.float32) for _ in specs
            )
        ),
    )
    nan = float("nan")
    synthetic = torch.tensor(
        [
            [5.0, nan, nan, nan, nan, nan],
            [1.0, 7.0, nan, nan, nan, nan],
            [3.0, 3.0, 9.0, nan, nan, nan],
        ]
    )
    monkeypatch.setattr(
        sparse_indexer, "fp8_fp4_mqa_logits", lambda *a, **kw: synthetic
    )

    def fake_topk(logits, ks, ke, out, num_rows, s0, s1, topk):
        for r in range(num_rows):
            n = int(ke[r]) - int(ks[r])
            order = torch.argsort(logits[r, :n], descending=True, stable=True)
            out[r, : min(n, topk)] = order[:topk].to(torch.int32)

    monkeypatch.setattr(sparse_indexer.ops, "top_k_per_row_prefill", fake_topk)

    session = dsa_trace.TraceSession(
        str(tmp_path),
        run_id=1,
        tp_rank=0,
        tp_world_size=1,
        k=k,
        device="cpu",
        pin_memory=False,
        ring_slots=2,
        capacity_rows=8,
    )
    nodes = session.ledger.add_path([10, 11, 12])
    ctx = dsa_trace.TraceContext(
        run_id=1,
        step_id=0,
        tp_rank=0,
        tp_world_size=1,
        rows=[dsa_trace.RowMeta(1, nodes[p], p, 10 + p) for p in range(rows)],
    )
    live = torch.full((rows, k), 17, dtype=torch.int32)
    dsa_trace.set_active(session, ctx)
    try:
        result = sparse_indexer.sparse_attn_indexer(
            torch.full((rows, 1), float("inf")),
            INDEXER_LAYER,
            torch.empty(1),
            torch.full((rows, 1), float("inf")),
            None,
            torch.arange(rows * 4, dtype=torch.float32).reshape(rows, 4),
            torch.full((rows, 1), float("inf")),
            128,
            "ue8m0",
            k,
            4,
            4096,
            4096,
            live,
            False,
            False,
            MLA_LAYER,
        )
    finally:
        dsa_trace.clear_active()
    session.close()

    assert result is live
    assert torch.all(live == 17)  # live selection untouched by shadow scoring
    rec = dsa_trace.read_records(session.records_path, k)
    hdr = rec["header"]
    assert len(rec) == rows
    assert all(hdr["flags"] & dsa_trace.Flag.SHADOW)
    assert not any(hdr["flags"] & dsa_trace.Flag.VIOLATION)
    assert hdr["layer_id"].tolist() == [0, 0, 0]
    assert hdr["valid_count"].tolist() == [1, 2, 3]
    assert rec["ids"][2, :3].tolist() == [2, 0, 1]  # stable tie order preserved
    assert rec["scores"][2, :3].tolist() == [9.0, 3.0, 3.0]
    assert all(math.isnan(t) for t in hdr["tau"])  # every prefix shorter than k


def test_sparse_attn_indexer_keeps_eager_break_decorator() -> None:
    """tollbooth: the shadow-scoring helper was inserted between
    ``@eager_break_during_capture`` and ``def sparse_attn_indexer`` and silently
    took the decorator with it. The break point must stay on the custom-op
    kernel (the helper runs inside it, in the same eager segment)."""
    import ast
    import inspect

    tree = ast.parse(inspect.getsource(sparse_indexer))
    decorators = {
        node.name: [ast.unparse(d) for d in node.decorator_list]
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
    }
    assert "eager_break_during_capture" in decorators["sparse_attn_indexer"]
    assert "eager_break_during_capture" not in decorators["_shadow_prefill_capture"]
