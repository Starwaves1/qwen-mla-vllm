# int8-tok decode microbench (kb2 methodology: CUDA events, median of 15, no L2 flush)
# env: RANKS, CS, CTXS, CFGS ';'-separated 'hb=8,bn=64,w=8,st=1,pv=0,sp=0' (sp=0 -> auto)
import os, sys, math, torch
sys.path.insert(0, '/home/garrett/tools/mla-vllm/src/qwen-mla-vllm')
import qwen_mla.ops.mla_int8_decode as I
dev = torch.device('cuda'); H = 24; ROPE = 232
PAGE = {256: int(os.environ.get('P256', 1664)), 768: int(os.environ.get('P768', 1664)), 1792: int(os.environ.get('P1792', 832))}
sms = torch.cuda.get_device_properties(0).multi_processor_count
ranks = [int(x) for x in os.environ.get('RANKS', '768').split(',')]
cs = [int(x) for x in os.environ.get('CS', '1,4').split(',')]
ctxs = [int(x) for x in os.environ.get('CTXS', '8192,65536').split(',')]
cfgs = [dict(kv.split('=') for kv in c.split(',')) for c in os.environ.get('CFGS', 'hb=8').split(';')]
WAVES = float(os.environ.get('WAVES', '2'))


def auto_split(c, nhb, ctx, bn):
    s = math.ceil(WAVES * sms / (c * nhb))
    return max(1, min(s, ctx // (bn * 4)))


def timeit(fn, n=15):
    fn(); torch.cuda.synchronize(); ts = []
    for _ in range(n):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record(); fn(); b.record(); b.synchronize(); ts.append(a.elapsed_time(b))
    return sorted(ts)[n // 2]


for rank in ranks:
    page = PAGE[rank]; pps = -(-max(ctxs) // page); nseq = max(cs)
    NP = nseq * pps
    pb = I.page_bytes(rank, page)
    cache = torch.randint(-127, 128, (NP, pb), dtype=torch.int8, device=dev).view(torch.uint8)
    so = page * (rank + 256)
    cache[:, so:so + 4 * page] = torch.full((NP, 2 * page), 0.05, dtype=torch.float16, device=dev).view(torch.uint8)
    cache[:, so + 4 * page:] = 128
    perm = torch.randperm(NP, generator=torch.Generator().manual_seed(0)).to(torch.int32).to(dev)
    for c in cs:
        for ctx in ctxs:
            bt = perm.view(nseq, pps)[:c].contiguous()
            lens = torch.full((c,), ctx, dtype=torch.int32, device=dev)
            torch.manual_seed(1)
            q = (torch.randn(c * 4, H, rank + ROPE, device=dev) * 0.05).to(torch.bfloat16)
            q8, qs, qr8, qrs = I.prep_q(q, rank, 1 / 16.0)
            out = torch.empty(c * 4, H, rank, dtype=torch.bfloat16, device=dev)
            lse = torch.empty(c * 4, H, dtype=torch.float32, device=dev)
            line = []
            for cfg in cfgs:
                hb = int(cfg.get('hb', 8)); bn = int(cfg.get('bn', 64)); w = int(cfg.get('w', 8)); st = int(cfg.get('st', 1))
                pv = int(cfg.get('pv', 0)); sp = int(cfg.get('sp', 0))
                extra = {k: int(v) for k, v in cfg.items() if k not in ('hb','bn','w','st','pv','sp')}
                nhb = -(-H // hb)
                ns = sp or auto_split(c, nhb, ctx, bn)
                opart = torch.empty(c * H * 4 * ns * rank, dtype=torch.float32, device=dev)
                lsep = torch.empty(c * H * 4 * ns, dtype=torch.float32, device=dev)
                if os.environ.get('QP'):
                    sys.path.insert(0, '/tmp/mlamx/int8'); import qprep
                    qq = (q8, qs, qr8, qrs)
                    fn = lambda: (qprep.prep_q(q, rank, 1 / 16.0, out=qq), I.int8_decode(q8, qs, qr8, qrs, cache, bt, lens, rank, page, ns, out, lse, hb=hb, bn=bn,
                                           pv_int8=pv, num_warps=w, num_stages=st, opart=opart, lsepart=lsep, **extra))
                else:
                  fn = lambda: I.int8_decode(q8, qs, qr8, qrs, cache, bt, lens, rank, page, ns, out, lse, hb=hb, bn=bn,
                                           pv_int8=pv, num_warps=w, num_stages=st, opart=opart, lsepart=lsep, **extra)
                try:
                    ms = timeit(fn)
                    ns_tok = ms * 1e6 / (c * ctx)
                    line.append(f"{','.join(f'{k}={v}' for k, v in cfg.items())},sp={ns}: {ms:.3f} ms ({ns_tok:.2f} ns/tok)")
                except Exception as ex:
                    line.append(f"{','.join(f'{k}={v}' for k, v in cfg.items())}: {type(ex).__name__} {str(ex)[:80]}")
            print(f'rank {rank} c={c} ctx {ctx} | ' + ' | '.join(line), flush=True)
    del cache; torch.cuda.empty_cache()
