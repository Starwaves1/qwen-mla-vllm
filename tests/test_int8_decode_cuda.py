# CUDA int8-tok decode (qwen_mla/csrc/mla_int8_decode.cu) correctness:
#   scores (mode 0) vs exact on the dequantized cache; outputs vs exact attention on the same quantized inputs (arith)
#   and on fp32 latents (true), next to the Triton int8 P kernel and today's FP8 kernel; lens 1..4, empty splits,
#   page boundaries; ROT=1 applies the block-Hadamard rotation (needed at r1792 for the peaky case).
# env: R (rank, default all), NM (row tiles per CTA for r256/r768), W (score / full / all), ROT, FP8
import os, sys, math, torch
sys.path.insert(0, '/home/garrett/tools/mla-vllm/src/qwen-mla-vllm'); sys.path.insert(0, '/home/garrett/tools/mla-vllm/src/qwen-mla-vllm/tests')
import qwen_mla.ops.mla_int8_decode as I
import qwen_mla.ops.mla_int8_cuda as CU
import test_int8_decode as T
dev = T.dev; H = 24; SM = T.SM

def run(rank, page, Ls, kind='real', nsplit=4, bn=None, mode=1, seed=0, fp8=False, rot=0):
    bn = bn or int(os.environ.get('BN', 16 if rank == 1792 else 32))
    NMV = 1 if rank == 1792 else int(os.environ.get('NM', 3))
    ctx = max(Ls); nreq = len(Ls)
    lat, rp, rms, q, bt = T.make(nreq, ctx, rank, page, seed + rank, kind)
    if rot:
        Hb = T.hadamard_blocks(rank); lat_k = lat @ Hb; q_k = q.float().clone(); q_k[..., :rank] = q_k[..., :rank] @ Hb; q_k = q_k.to(torch.bfloat16)
    else:
        lat_k, q_k = lat, q
    cache = I.pack_cache(lat_k, rp, rms)
    lens = torch.tensor(Ls, dtype=torch.int32, device=dev)
    lens_rows = torch.tensor([L - 3 + j for L in Ls for j in range(4)], dtype=torch.int32, device=dev)
    btr = bt.repeat_interleave(4, 0).contiguous()
    dl, dr, drms = I.unpack_cache(cache, rank, page)
    q8, qs, qr8, qrs = I.prep_q(q_k, rank, SM)
    qd = T.deq_q(q8, qs, qr8, qrs, rank)
    if mode == 0:
        sd = torch.full((nreq * 96, ctx), float('nan'), device=dev)
        CU.int8_decode_cuda(q8, qs, qr8, qrs, cache, bt, lens, rank, page, nsplit, None, None, bn=bn, mode=0, sdbg=sd, nm=NMV)
        torch.cuda.synchronize()
        err = 0.0; mx = 0.0
        for r in range(nreq):
            L = Ls[r]
            c = dl[bt[r].long()].reshape(-1, rank)[:L].double(); rr = dr[bt[r].long()].reshape(-1, 232)[:L].double()
            mm = drms[bt[r].long()].reshape(-1, H)[:L].double()
            for j in range(4):
                n = L - 3 + j
                if n <= 0: continue
                qq = qd[4 * r + j].double()                                   # [24, rank+232]
                s = (qq[:, :rank] @ c[:n].T + qq[:, rank:] @ rr[:n].T) / mm[:n].T * SM * I.LOG2E   # [24, n]
                rows = (r * 24 + torch.arange(24, device=dev)) * 4 + j
                got = sd[rows, :n].double()
                err = max(err, (got - s).abs().max().item()); mx = max(mx, s.abs().max().item())
                if torch.isnan(got).any(): print('NaN/unwritten scores at', r, j, int(torch.isnan(got).sum()))
        return dict(score_abs_err=err, score_max=mx)
    out = torch.empty(nreq * 4, H, rank, dtype=torch.float32, device=dev)
    lse = torch.empty(nreq * 4, H, dtype=torch.float32, device=dev)
    CU.int8_decode_cuda(q8, qs, qr8, qrs, cache, bt, lens, rank, page, nsplit, out, lse, bn=bn, nm=NMV)
    oT = torch.empty_like(out); lT = torch.empty_like(lse)
    I.int8_decode(q8, qs, qr8, qrs, cache, bt, lens, rank, page, nsplit, oT, lT, hb=8, bn={768: 64, 256: 128, 1792: 32}[rank], pv_int8=True, acc16=True, qin=rank == 1792)
    r_arith = T.ref(qd, dl, dr, drms, btr, lens_rows, rank, page)
    o, oT_ = out, oT
    if rot:
        o, oT_ = out @ Hb.T, oT @ Hb.T; r_arith = (r_arith.float() @ Hb.T).double()
    rms_t = torch.exp2(-3.0 + ((rms.log2() + 3.0) * (255.0 / 6.0)).round().clamp(0, 255) * (6.0 / 255.0))
    r_true = T.ref(q.float(), lat, rp, rms_t, btr, lens_rows, rank, page)
    # lse check vs exact (natural log) on the quantized inputs
    res = dict(cu_arith=T.rel(o, r_arith), cu_true=T.rel(o, r_true), tri_arith=T.rel(oT_, r_arith), tri_true=T.rel(oT_, r_true),
               lse_diff=(lse - lT).abs().nan_to_num(0).max().item(),
               nan=bool(torch.isnan(out).any() or torch.isinf(out).any()))
    if fp8:
        of, fdl, fdr, frms = T.fp8_run(q, lat, rp, rms, btr, lens_rows, rank, page)
        res['fp8_true'] = T.rel(of, r_true)
    return res

def main(rank, what):
    page = 1664 if rank != 1792 else 832
    if what in ('score', 'all'):
        for Ls, ns in (([5000, 1700], 4), ([1, 2, 3, 4], 2), ([40, 100], 7)):
            print(rank, Ls, ns, run(rank, page, Ls, nsplit=ns, mode=0), flush=True)
    if what in ('full', 'all'):
        rot = int(os.environ.get('ROT', 0))
        for kind in ('real', 'diffuse', 'peaky'):
            r = run(rank, page, [5000, 1700], kind, nsplit=4, fp8=bool(int(os.environ.get('FP8', 1))), rot=rot)
            print(f'rot={rot} {rank} {kind} ' + ' '.join(f'{k}={v:.2e}' if isinstance(v, float) else f'{k}={v}' for k, v in r.items()), flush=True)
        for Ls, ns in (([1, 2, 3, 4], 2), ([40, 100, 3, 1700], 13), ([1664, 1665, 3328, 33], 5)):
            r = run(rank, page, Ls, 'real', nsplit=ns, fp8=False, rot=rot)
            print(f'edge {Ls} ns={ns} ' + ' '.join(f'{k}={v:.2e}' if isinstance(v, float) else f'{k}={v}' for k, v in r.items()), flush=True)


if __name__ == '__main__':
    for rank in [int(x) for x in os.environ.get('R', '768,256,1792').split(',')]:
        main(rank, os.environ.get('W', 'all'))
