"""int8-tok cache as the plugin drives it (QWEN_MLA_INT8_KV): cache write through a slot mapping into vLLM-shaped
[blocks, page_tokens, row] caches with a padded block stride, decode through QwenMLATritonImpl._forward_mqa_int8
(per-request block table, 1..4 query rows per request, zero-length padding requests), and the chunked-prefill
gather. Reference: fp32 attention on the dequantized cache (I.unpack_cache) with unquantized queries.

    python tests/test_int8_vllm_path.py
"""
import types

import torch

from qwen_mla.mla_impl import QwenMLATritonImpl, _int8_pages
from qwen_mla.ops import mla_int8_decode as I
from qwen_mla.ops.mla_int8_write import mla_int8_cache_write, mla_int8_gather


def run(rank, page, row, lens, qn, seed=0, dev="cuda"):
    g = torch.Generator(device=dev).manual_seed(seed)
    nreq, T = len(lens), sum(lens)
    per = max(-(-max(L, 1) // page) for L in lens) + 1
    nblk = nreq * per + 1
    stride = page * row + 4096                                   # padded page, as vLLM's page unification can make
    raw = torch.full((nblk * stride,), 0xFF, dtype=torch.uint8, device=dev)   # poison: unwritten bytes are NaN scales
    kv = torch.as_strided(raw, (nblk, page, row), (stride, row, 1)).view(torch.float8_e4m3fn)
    bt = (torch.randperm(nblk - 1, generator=g, device=dev)[: nreq * per] + 1).view(nreq, per).int().contiguous()

    lat = torch.randn(T, rank, generator=g, device=dev) * 0.4
    lat[:, :8] *= 6                                               # a few outlier channels
    rope = torch.randn(T, 232, generator=g, device=dev)
    rms = torch.rand(T, 24, generator=g, device=dev) * 1.5 + 0.4
    k_pe = torch.cat([rope, rms, torch.zeros(T, row - rank - 256, device=dev)], -1).to(torch.bfloat16)
    req = torch.cat([torch.full((L,), i, device=dev) for i, L in enumerate(lens)]).long()
    tok = torch.cat([torch.arange(L, device=dev) for L in lens]).long()
    slots = bt[req, tok // page].long() * page + tok % page
    slots = torch.cat([slots, torch.full((5,), -1, device=dev, dtype=torch.long)])   # CUDA-graph padding slots
    kv_c = torch.cat([lat, torch.randn(5, rank, device=dev)]).to(torch.bfloat16)
    k_pe = torch.cat([k_pe, torch.randn(5, k_pe.shape[1], device=dev).to(torch.bfloat16)])
    mla_int8_cache_write(kv_c, k_pe, _int8_pages(kv), slots, page)

    # dequantized cache per request
    pages = _int8_pages(kv)[:, : I.page_bytes(rank, page)]
    deq = []
    for i, L in enumerate(lens):
        a, b, c = I.unpack_cache(pages[bt[i].long()].contiguous(), rank, page)
        deq.append((a.reshape(-1, rank)[:L], b.reshape(-1, 232)[:L], c.reshape(-1, 24)[:L]))

    # decode
    sm = 256 ** -0.5
    declared = row - rank
    q_lat = torch.randn(nreq * qn, 24, rank, generator=g, device=dev)
    q_pe = torch.zeros(nreq * qn, 24, declared, device=dev)
    grp = torch.arange(24, device=dev) // 6
    qr = torch.randn(nreq * qn, 24, 58, generator=g, device=dev) * 2
    for h in range(24):
        q_pe[:, h, grp[h] * 58: grp[h] * 58 + 58] = qr[:, h]
    fake = types.SimpleNamespace(kv_lora_rank=rank, scale=sm)
    md = types.SimpleNamespace(causal=True, decode=types.SimpleNamespace(
        _mla_req_bt=bt, _mla_req_sl=torch.tensor(lens, dtype=torch.int32, device=dev)))
    o, lse = QwenMLATritonImpl._forward_mqa_int8(fake, (q_lat.to(torch.bfloat16), q_pe.to(torch.bfloat16)), kv, md)
    assert o.shape == (nreq * qn, 24, rank) and lse.shape == (nreq * qn, 24)
    worst = 0.0
    for i, L in enumerate(lens):
        c, rp, rm = deq[i]
        for k in range(qn):
            n = L - (qn - 1 - k)                                  # causal: the request's k-th query row
            if n <= 0:
                continue
            b = i * qn + k
            qb = q_lat[b].to(torch.bfloat16).float()
            qrb = qr[b].to(torch.bfloat16).float()
            s = qb @ c[:n].T + torch.einsum("hd,thd->ht", qrb, rp[:n].view(n, 4, 58)[:, grp])
            s = s / rm[:n].T * sm
            ref = torch.softmax(s, -1) @ c[:n]
            err = ((o[b].float() - ref).abs().max() / ref.abs().max()).item()
            worst = max(worst, err)
    assert torch.isfinite(o).all(), "non-finite output"

    # chunked-prefill gather (context starting at seq_starts, dst rows in the BF16 layout)
    starts = torch.tensor([min(7, L) for L in lens], dtype=torch.int32, device=dev)
    cnt = [L - min(7, L) for L in lens]
    cu = torch.tensor([0] + list(torch.tensor(cnt).cumsum(0)), dtype=torch.int32, device=dev)
    t2s = torch.cat([torch.full((n,), i, dtype=torch.int32, device=dev) for i, n in enumerate(cnt)])
    dst = torch.zeros(sum(cnt) + 3, row, dtype=torch.bfloat16, device=dev)
    mla_int8_gather(src_cache=kv, dst=dst, block_table=bt, cu_seq_lens=cu, token_to_seq=t2s, num_tokens=sum(cnt),
                    kv_cache_dtype="fp8", scale=None, seq_starts=starts, rank=rank, pages_of=_int8_pages)
    want = torch.cat([torch.cat([c[7:], rp[7:], rm[7:]], -1) for (c, rp, rm), L in zip(deq, lens) if L > 7])
    gerr = ((dst[: sum(cnt), : rank + 256].float() - want).abs() / (want.abs() + 1e-2)).max().item()
    print(f"rank {rank} page {page} row {row} lens {lens} qn {qn}: decode max rel err {worst:.2e}, gather {gerr:.2e}")
    assert worst < 5e-2, worst          # int8 q and int8 P (P4: kernel arithmetic alone up to ~1e-2)
    assert gerr < 1e-2, gerr


if __name__ == "__main__":
    run(768, 1664, 1056, [5000, 1700, 0], 4)
    run(768, 1664, 1056, [3, 2000], 2)
    run(256, 1664, 1056, [2500, 1], 1)
    run(1792, 832, 2112, [3000, 4, 0, 900], 4)
    run(1792, 832, 2112, [1700], 3)
    print("OK")
