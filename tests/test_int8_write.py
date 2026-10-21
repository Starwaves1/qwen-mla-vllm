# int8-tok cache writer vs the torch packer (byte-exact), through a shuffled slot mapping with padded (-1) slots
import sys
import torch
sys.path.insert(0, '/home/garrett/tools/mla-vllm/src/qwen-mla-vllm')
import qwen_mla.ops.mla_int8_decode as I
from qwen_mla.ops.mla_int8_write import mla_int8_cache_write

dev = torch.device('cuda')
for rank, page in ((256, 1664), (768, 1664), (1792, 832)):
    g = torch.Generator(device=dev).manual_seed(rank)
    NP = 3
    lat = torch.randn(NP, page, rank, device=dev, generator=g) * 2.0
    lat[..., :4] *= 8
    lat[1, 5] = 0.0                                          # all-zero token: scale falls back to 1
    rp = torch.randn(NP, page, 232, device=dev, generator=g) * 1.5
    rms = torch.exp(torch.randn(NP, page, 24, device=dev, generator=g) * 0.6)
    lat, rp, rms = lat.bfloat16().float(), rp.bfloat16().float(), rms.bfloat16().float()   # what vLLM hands over
    ref = I.pack_cache(lat, rp, rms)
    T = NP * page
    perm = torch.randperm(T, generator=torch.Generator().manual_seed(1)).to(dev)
    kv_c = lat.reshape(T, rank)[perm].bfloat16()
    k_pe = torch.cat([rp.reshape(T, 232), rms.reshape(T, 24), torch.zeros(T, 8, device=dev)], -1)[perm].bfloat16()
    slots = perm.to(torch.int64)
    pad = torch.full((7,), -1, dtype=torch.int64, device=dev)
    kv_c = torch.cat([kv_c, torch.zeros(7, rank, device=dev, dtype=kv_c.dtype)])
    k_pe = torch.cat([k_pe, torch.zeros(7, k_pe.shape[1], device=dev, dtype=k_pe.dtype)])
    slots = torch.cat([slots, pad])
    out = torch.zeros_like(ref)
    mla_int8_cache_write(kv_c, k_pe, out, slots, page)
    diff = (out != ref)
    n = int(diff.sum())
    regions = []
    if n:
        o = 0
        for name, width in (('latent', page * rank), ('rope', page * 256), ('sigma', 2 * page), ('rho', 2 * page), ('rms', 24 * page)):
            regions.append(f'{name}:{int(diff[:, o:o + width].sum())}')
            o += width
    print(f'r{rank}: {n} differing bytes of {ref.numel()}' + (f' ({" ".join(regions)})' if n else ''), flush=True)
