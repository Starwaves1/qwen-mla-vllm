# int8-tok decode bench (env R=rank, CS, CTXS). Final grid of 2026-10-21: today's FP8 two-pass (prod tiles) vs Triton int8 (P2 configs) vs CUDA int8, measured interleaved per cell
# ms per layer call (c requests x 4 MTP rows, 24 heads), median of 15, sync per call; int8 paths include q prep + reduce
import os, sys, torch
sys.path.insert(0, '/home/garrett/tools/mla-vllm/src/qwen-mla-vllm')
for k_, v_ in dict(QWEN_MLA_2P_MIN_LV='512', QWEN_MLA_2P_BN='64', QWEN_MLA_2P_DC='128', QWEN_MLA_2P_DVC='256', QWEN_MLA_2P_STAGES_A='1', QWEN_MLA_2P_STAGES_B='1').items():
    os.environ[k_] = v_
import qwen_mla.ops.mla_decode_attention as M
import qwen_mla.ops.mla_int8_decode as I
import qwen_mla.ops.mla_int8_cuda as CU
from qwen_mla.mla_impl import splitk_cap
from vllm.v1.attention.backends.mla.triton_mla import _compute_num_kv_splits
dev = torch.device('cuda'); H, ROPE, K = 24, 232, 3; F8 = torch.float8_e4m3fn; SMS = 82
rank = int(os.environ['R']); cs = [int(x) for x in os.environ.get('CS', '1,2,4').split(',')]
ctxs = [int(x) for x in os.environ.get('CTXS', '8192,32768,65536,131072').split(',')]
P8 = {256: 3328, 768: 1664, 1792: 832}[rank]; PI = {256: 1664, 768: 1664, 1792: 832}[rank]
TRI = {256: dict(hb=8, bn=128, acc16=True), 768: dict(hb=8, bn=64, acc16=True), 1792: dict(hb=8, bn=32, acc16=True, qin=True)}[rank]
CUC = {256: [(3, 32), (2, 32)], 768: [(3, 32), (2, 32)], 1792: [(1, 16)]}[rank]
def timeit(fn, n=15):
    fn(); torch.cuda.synchronize(); ts = []
    for _ in range(n):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record(); fn(); b.record(); b.synchronize(); ts.append(a.elapsed_time(b))
    return sorted(ts)[n // 2]
nseq = max(cs); ks = torch.tensor(1.0, device=dev)
w = rank + ROPE + H; w += (-w) % 16
pps8 = -(-max(ctxs) // P8); c8 = (torch.randn(nseq * pps8, P8, 1, w, device=dev, dtype=torch.bfloat16) * 0.5).to(F8).view(torch.uint8)
c8[..., rank + ROPE:rank + ROPE + H] = 128; c8 = c8.view(F8)
perm8 = torch.randperm(nseq * pps8, generator=torch.Generator().manual_seed(0)).to(torch.int32).to(dev)
ppsi = -(-max(ctxs) // PI); pbi = I.page_bytes(rank, PI)
ci = torch.randint(-127, 128, (nseq * ppsi, pbi), dtype=torch.int8, device=dev).view(torch.uint8)
so = PI * (rank + 256); ci[:, so:so + 4 * PI] = torch.full((nseq * ppsi, 2 * PI), 0.05, dtype=torch.float16, device=dev).view(torch.uint8); ci[:, so + 4 * PI:] = 128
permi = torch.randperm(nseq * ppsi, generator=torch.Generator().manual_seed(0)).to(torch.int32).to(dev)
for c in cs:
    for ctx in ctxs:
        B = c * (K + 1); seq = torch.arange(c, device=dev).repeat_interleave(K + 1)
        bt8 = perm8.view(nseq, pps8)[seq].contiguous(); lens8 = (ctx - K + torch.arange(K + 1, device=dev).repeat(c)).to(torch.int32)
        spl = splitk_cap(B, H, rank + 1, _compute_num_kv_splits(ctx, SMS))
        torch.manual_seed(1); q = (torch.randn(B, H, rank + ROPE, device=dev) * 0.05).to(torch.bfloat16)
        o = torch.empty(B, H, rank, dtype=torch.bfloat16, device=dev); lse = torch.empty(B, H, dtype=torch.float32, device=dev)
        lg = torch.empty(B, H, spl, rank + 1, dtype=torch.float32, device=dev)
        t_today = timeit(lambda: M.decode_attention_fwd(q, c8, c8[..., :rank], o, lse, bt8, lens8, lg, spl, 1 / 16.0, P8, k_scale=ks, v_scale=ks,
                                                        is_mla=True, rms_offset=rank + ROPE, rms_log8=True))
        bti = permi.view(nseq, ppsi)[:c].contiguous(); lensi = torch.full((c,), ctx, dtype=torch.int32, device=dev)
        qq = I.prep_q(q, rank, 1 / 16.0)
        tri = []
        for sp in (13, 27, 54):
            op = torch.empty(c * 96 * sp * rank, device=dev); lp = torch.empty(c * 96 * sp, device=dev)
            tri.append((timeit(lambda: (I.prep_q(q, rank, 1 / 16.0, out=qq), I.int8_decode(*qq, ci, bti, lensi, rank, PI, sp, o, lse, pv_int8=True,
                                                                                          num_warps=8, opart=op, lsepart=lp, **TRI))), sp))
        cu = []
        for nm, bn in CUC:
            nhb = 24 // (4 * nm)
            for wv in (1, 2, 3):
                sp = max(1, min(SMS * wv // (nhb * c), ctx // 256))
                op = torch.empty(c * 96 * sp * rank, dtype=torch.float16, device=dev); lp = torch.empty(c * 96 * sp, device=dev)
                cu.append((timeit(lambda: (I.prep_q(q, rank, 1 / 16.0, out=qq), CU.int8_decode_cuda(*qq, ci, bti, lensi, rank, PI, sp, o, lse, bn=bn,
                                                                                                    nm=nm, opart=op, lsepart=lp))), nm, sp))
        tb, cb = min(tri), min(cu)
        print(f'r{rank} c={c} ctx={ctx}: today {t_today:.3f} | triton {tb[0]:.3f} (sp {tb[1]}) | cuda {cb[0]:.3f} (nm {cb[1]} sp {cb[2]}) | '
              f'x{t_today / cb[0]:.2f} vs today, x{tb[0] / cb[0]:.2f} vs triton', flush=True)
