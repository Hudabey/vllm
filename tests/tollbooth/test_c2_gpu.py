# SPDX-License-Identifier: Apache-2.0
"""tollbooth C2 focused GPU tests (analysis/CAPTURE_BRIEF.md): real DeepGEMM logits
kernels, real vLLM top-k dispatch, real indexer K-cache insert, through
``sparse_attn_indexer()`` exactly like the gate. No model, no replay.

  test_c2_streams_and_full_row_ownership   S2 bytes == kernel inputs; S4 file == the
                                           live logits row of that layer (spy clone) even
                                           after the next layer ran; allocator reuse of
                                           the logits buffer is recorded as a fact
  test_c2_compacted_block_table            M3: the paged kernel called with a block table
                                           listing only a subset of the request's pages
                                           gives, per key, bitwise the same fp32 score as
                                           the full-table call (sampled evidence, recorded)
  test_c2_full_row_determinism             M2: two identical forwards give bitwise-equal
                                           full fp32 rows
  test_c2_page_layout_and_appended_key     S3 through the real insert kernel: page layout
                                           (64 x 128 e4m3 values, then 64 fp32 scales),
                                           dequantized key within e4m3 tolerance of the
                                           inserted bf16 key, C2 snapshot/appended bytes

Facts are appended to ``$TOLLBOOTH_GATE_OUT/gate-results.json`` under ``c2_*`` keys.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pytest
import torch

from vllm import _custom_ops as ops
from vllm.model_executor.layers import dsa_capture_ext as ext
from vllm.model_executor.layers import dsa_trace
from vllm.model_executor.layers import sparse_attn_indexer as sparse_indexer
from vllm.utils.deep_gemm import get_paged_mqa_logits_metadata

from tests.tollbooth.test_gpu_gate import (  # noqa: E402
    BLOCK, BLOCK_TABLE_WIDTH, CACHE_ROW, HEAD_DIM, K, MAX_MODEL_LEN, N_HEADS, QUANT_BLOCK,
    SCALE_FMT, LogitsSpy, env, make_batch, new_session, record_fact, run_layer, trace_dir,
)

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA only"),
]

_ = env  # re-exported fixture


def _hashes(batch):
    out = {}
    for r in batch.rows:
        out.setdefault(r.request_key, r.prompt_hash)
    return out


def test_c2_streams_and_full_row_ownership(env, tmp_path_factory):
    device = env.device
    L = 4096
    tdir = trace_dir(tmp_path_factory, "c2-streams")
    session = new_session(tdir, device, sample_mode=dsa_trace.SampleMode.WINDOWED, windows={})
    batch = make_batch(env, [L], [L], seed=700, session=session)
    hashes = _hashes(batch)
    dec_key, pre_key = batch.rows[0].request_key, batch.rows[-1].request_key
    dec_h, pre_h = hashes[dec_key], hashes[pre_key]
    session.windows = {dec_h: [(L - 1, L - 1)], pre_h: [(4000, 4095)]}
    c2 = ext.C2Session(session, device=device, max_model_len=MAX_MODEL_LEN, queries=True,
                       rows_spec={dec_h: {L - 1}, pre_h: {4090}}, num_slots=8)
    T = batch.num_tokens
    buf = torch.full((T, K), -1, dtype=torch.int32, device=device)
    ctx = dsa_trace.TraceContext(run_id=1, step_id=0, tp_rank=0, tp_world_size=1, rows=batch.rows)
    ptrs = {}
    with LogitsSpy() as spy:
        for layer in (0, 1):
            kv = batch.layers[layer].kv_cache_base.clone()
            n0 = len(spy.calls)
            run_layer(env, batch, layer, kv, buf, (session, ctx))
            ptrs[layer] = [c.ptr for c in spy.calls[n0:]]
        calls = spy.calls[:]
    c2.flush(); c2.close(); session.close()
    torch.cuda.synchronize()
    rec = dsa_trace.read_records(session.records_path, K)
    qr = ext.read_query_records(c2.queries_path)
    assert len(qr) == len(rec) == 2 * 97
    h = qr["header"]
    for i in range(len(qr)):
        layer = int(h["layer_id"][i]); pos = int(h["query_position"][i])
        inp = batch.layers[layer]
        row = 0 if int(h["request_key"][i]) == dec_key else 1 + pos  # token row in the forward
        assert np.array_equal(qr["q"][i], inp.q_fp8[row].view(torch.uint8).reshape(-1).cpu().numpy())
        assert np.array_equal(qr["w"][i], inp.weights[row].cpu().numpy())
    # S4: the retained rows equal the spy's clones of the live logits of THAT layer
    m = json.load(open(c2.manifest_path))
    assert len(m["rows"]) == 4
    per_layer = {0: calls[: len(ptrs[0])], 1: calls[len(ptrs[0]) :]}
    for e in m["rows"]:
        got = np.fromfile(os.path.join(tdir, e["file"]), dtype=np.float32)
        layer = e["layer_id"]
        if e["request_key"] == dec_key:
            call = [c for c in per_layer[layer] if c.kind == "decode"][0]
            exp = call.logits[0, : L].cpu().numpy()
        else:
            exp = None
            for c in per_layer[layer]:
                if c.kind != "prefill":
                    continue
                P = (c.ke - c.ks).cpu().numpy()
                hit = np.flatnonzero(P == e["prefix_length"])
                if len(hit):
                    r = int(hit[0]); ks = int(c.ks[r])
                    exp = c.logits[r, ks : ks + e["prefix_length"]].cpu().numpy()
                    break
            assert exp is not None
        assert len(got) == e["prefix_length"] and np.array_equal(got.view(np.uint32), exp.view(np.uint32)), e["file"]
    reuse = {f"layer{l}": ptrs[l] for l in ptrs}
    reuse["decode_logits_buffer_reused_by_next_layer"] = bool(
        [c.ptr for c in per_layer[0] if c.kind == "decode"] == [c.ptr for c in per_layer[1] if c.kind == "decode"])
    assert all(len(s.device_refs) == 0 for s in c2.ring.slots)
    record_fact("c2_streams_and_ownership", {"queries": len(qr), "rows": len(m["rows"]),
                                            "rows_bytes": m["stats"]["rows_bytes"], "buffer_reuse": reuse,
                                            "ring_stalls": m["ring_stalls"]})


@pytest.mark.parametrize("L", (4096, 8192))
def test_c2_compacted_block_table(env, tmp_path_factory, L):
    """M3. The full-table decode call vs compacted block tables (subsets of the
    request's pages, whole pages only). Sampled evidence, not a proof."""
    device = env.device
    batch = make_batch(env, [L], [], seed=710 + L, logits_mb="512")
    buf = torch.full((1, K), -1, dtype=torch.int32, device=device)
    kv = batch.layers[0].kv_cache_base.clone()
    run_layer(env, batch, 0, kv, buf, None)  # inserts the decode token's key
    inp = batch.layers[0]
    dec = batch.metadata.decode
    kvv = sparse_indexer.kv_cache_as_quant_view(kv, HEAD_DIM, False)
    q = inp.q_fp8.reshape(1, 1, N_HEADS, HEAD_DIM)
    w = inp.weights[:1]
    seq = dec.seq_lens[:1]
    if seq.dim() == 1:
        seq = seq.unsqueeze(-1)
    bt = dec.block_table[:1]
    full = sparse_indexer.fp8_fp4_paged_mqa_logits((q, None), kvv, w, seq, bt, dec.schedule_metadata,
                                                     max_model_len=MAX_MODEL_LEN, clean_logits=False)[0, :L].clone()
    nb = L // BLOCK
    assert L % BLOCK == 0
    num_sms = env.builder.num_sms
    g = torch.Generator().manual_seed(L)
    subsets = {
        "all": list(range(nb)),
        "first_half": list(range(nb // 2)),
        "random_quarter": sorted(torch.randperm(nb, generator=g)[: max(1, nb // 4)].tolist()),
        "single_last": [nb - 1],
        "reversed_all": list(range(nb))[::-1],
    }
    facts = {}
    for name, S in subsets.items():
        bt2 = torch.zeros_like(bt)
        bt2[0, : len(S)] = bt[0, S]
        seq2 = torch.tensor([[len(S) * BLOCK]], dtype=torch.int32, device=device)
        sched2 = get_paged_mqa_logits_metadata(seq2, BLOCK, num_sms)
        out = sparse_indexer.fp8_fp4_paged_mqa_logits((q, None), kvv, w, seq2, bt2, sched2,
                                                        max_model_len=MAX_MODEL_LEN, clean_logits=False)[0, : len(S) * BLOCK]
        exp = torch.cat([full[b * BLOCK : (b + 1) * BLOCK] for b in S])
        eq = out.view(torch.int32) == exp.view(torch.int32)
        n_bad = int((~eq).sum())
        max_ulp = 0
        if n_bad:
            max_ulp = int((out.view(torch.int32) - exp.view(torch.int32)).abs().max())
        facts[name] = {"pages": len(S), "keys": len(S) * BLOCK, "bitwise_mismatches": n_bad, "max_ulp_diff": max_ulp}
    record_fact(f"c2_compacted_block_table_L{L}", facts)
    bad = {k: v for k, v in facts.items() if v["bitwise_mismatches"]}
    assert not bad, f"compacted-table scores differ from the full-table call: {bad}"


def test_c2_full_row_determinism(env, tmp_path_factory):
    """M2 at kernel level: identical inputs, two forwards, bitwise-equal fp32 rows."""
    device = env.device
    L = 8192
    batch = make_batch(env, [L], [], seed=720, logits_mb="512")
    buf = torch.full((1, K), -1, dtype=torch.int32, device=device)
    rows = []
    for _ in range(2):
        kv = batch.layers[0].kv_cache_base.clone()
        with LogitsSpy() as spy:
            run_layer(env, batch, 0, kv, buf, None)
        rows.append(spy.calls[-1].logits[0, :L].clone())
    eq = torch.equal(rows[0].view(torch.int32), rows[1].view(torch.int32))
    record_fact("c2_full_row_determinism", {"L": L, "bitwise_equal": eq})
    assert eq


def test_c2_page_layout_and_appended_key(env, tmp_path_factory):
    device = env.device
    n_pos, nb = 130, 3
    gen = torch.Generator(device=device); gen.manual_seed(730)
    keys = torch.randn(n_pos, HEAD_DIM, generator=gen, device=device).to(torch.bfloat16)
    kv = torch.zeros(8, BLOCK, CACHE_ROW, dtype=torch.uint8, device=device)
    phys = torch.tensor([5, 1, 6], dtype=torch.int64)
    pos = torch.arange(n_pos)
    slots = (phys[pos // BLOCK] * BLOCK + pos % BLOCK).to(device)
    ops.indexer_k_quant_and_cache(keys, kv, slots, QUANT_BLOCK, SCALE_FMT)
    kv2d = kv.reshape(kv.shape[0], -1)
    assert kv2d.shape[1] == BLOCK * CACHE_ROW
    # layout: values at off*128, fp32 scale at BLOCK*128 + off*4; dequant within e4m3 tolerance
    page = kv2d[6].cpu()
    max_rel = 0.0
    for p in (128, 129):
        off = p % BLOCK
        vals = page[off * HEAD_DIM : (off + 1) * HEAD_DIM].view(torch.float8_e4m3fn).float()
        scale = page[BLOCK * HEAD_DIM + off * 4 : BLOCK * HEAD_DIM + (off + 1) * 4].view(torch.float32)[0]
        assert torch.isfinite(scale) and scale > 0
        ref = keys[p].float().cpu()
        deq = vals * scale
        rel = ((deq - ref).abs() / ref.abs().clamp_min(1e-2)).max().item()
        max_rel = max(max_rel, rel)
    assert max_rel < 0.15, max_rel  # e4m3 has 3 mantissa bits (~6 % step) plus the group scale
    # C2 S3 through the CacheAccess interface over the real cache tensor
    tdir = trace_dir(tmp_path_factory, "c2-pages")
    ph = "9" * 64
    session = new_session(tdir, device, sample_mode=dsa_trace.SampleMode.WINDOWED, windows={ph: [(128, 129)]})
    c2 = ext.C2Session(session, device=device, max_model_len=MAX_MODEL_LEN, queries=False, pages=True, num_slots=2)

    class Access(ext.CacheAccess):
        def block_table_row(self, request_key):
            return np.array([5, 1, 6, 0], dtype=np.int32)

        def layer_caches(self):
            return [(0, kv2d)]

    toks = list(range(1, n_pos + 1))
    for step, p in enumerate((128, 129)):
        nodes = session.ledger.add_path(toks)
        meta = [dsa_trace.RowMeta(77, nodes[p], p, toks[p], 0, dsa_trace.Phase.DECODE, ph)]
        ctx = dsa_trace.TraceContext(run_id=1, step_id=step, tp_rank=0, tp_world_size=1, rows=meta)
        c2.post_forward(ctx, Access())
    c2.close(); session.close()
    m = json.load(open(c2.manifest_path))
    A = m["pages"][ph]["layers"]["0"]["A"]
    got = np.fromfile(os.path.join(tdir, A["file"]), dtype=np.uint8).reshape(3, BLOCK * CACHE_ROW)
    assert np.array_equal(got, kv2d[[5, 1, 6]].cpu().numpy()) and A["positions_covered"] == 129
    raw = np.fromfile(os.path.join(tdir, "pages", f"{ph}.appended.records"), dtype=np.uint8)
    per = 64 + CACHE_ROW
    assert len(raw) == 2 * per
    for j, p in enumerate((128, 129)):
        off = p % BLOCK
        exp = np.concatenate([page[off * HEAD_DIM : (off + 1) * HEAD_DIM].numpy(),
                              page[BLOCK * HEAD_DIM + off * 4 : BLOCK * HEAD_DIM + (off + 1) * 4].numpy()])
        assert np.array_equal(raw[j * per + 64 : (j + 1) * per], exp)
    record_fact("c2_page_layout", {"max_rel_dequant_error": max_rel, "page_bytes": BLOCK * CACHE_ROW,
                                   "layout": "values[64x128] then scales[64 fp32]"})
