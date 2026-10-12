#!/usr/bin/env python3
"""GPU: FP8 KV cache on sm_86 -- correctness and decode speed vs a bf16 cache (Qwen3.8-27B MLA, TP=1 shapes).

Correctness (every QWEN_MLA_FP8_DECODE mode, through the real decode kernels):
  decode-V   1-token attention returns V exactly: output == float(e4m3) * k_scale, bit for bit, every finite code at
             every latent position (single-pass ranks 256/768, two-pass 1792)
  decode-K   one-hot q, rms 1, sm_scale 1: the LSE is the score q.k == float(e4m3) * k_scale, bit for bit, every
             latent and rope position
  attention  fp32 output vs exact attention over the dequantized cache (< 1e-3), and vs the bf16-KV reference:
             within the FP8 quantization error (|err - quant err| <= 0.1 * quant err + 1e-3)
  write      the one-launch cache write (ops/mla_fp8_write.py) writes the torch path's bytes (log8 and bf16 rms)
Speed (CUDA graphs, L2 flushed before each call, as between layers in a forward):
  decode attention per layer type (rank 256 / 768 / 1792) and per step (2 x 256 + 13 x 768 + 1 x 1792), bf16 vs FP8,
  MTP verify (k=3: 4 query rows per sequence), c = 1 and 3, contexts 8k / 64k / 128k / 190k
  cache write per layer for 4 / 12 / 8192 tokens: torch ops vs the Triton kernel vs vLLM's bf16 concat_and_cache_mla
PASS = all correctness checks, and with the default mode (asm) the FP8 per-step attention time <= 1.02 x bf16 at
every c in {1, 3} and context >= 64k, and the Triton cache write no slower than the torch ops at 4 and 12 tokens.

  python tests/gpu_fp8_bench.py --out DIR [--modes asm,hoist,triton] [--budget-s 840]
Writes DIR/fp8_bench.json and DIR/fp8_bench.txt; exit 0 PASS, 1 FAIL, 2 error. Peak GPU memory ~2.6 GB.
"""
import argparse, json, os, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
RANKS = {256: 2, 768: 13, 1792: 1}                     # rank -> layers of that rank (16 MLA layers)
ROPE, HEADS, K_MTP = 232, 24, 3
CTXS = (8192, 65536, 131072, 194560)
PAGE_FP8 = {256: 3328, 768: 1664, 1792: 832}           # block sizes vLLM picks (1.70 MB pages)
PAGE_BF16 = {256: 1664, 768: 832, 1792: 416}


# ----------------------------------------------------------------------------------------------- worker
def worker(mode, do_bf16, out_path, deadline):
    import torch
    sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
    import qwen_mla.ops.mla_decode_attention as M
    from qwen_mla.mla_impl import splitk_cap
    from vllm.v1.attention.backends.mla.triton_mla import _compute_num_kv_splits
    assert M._FP8_MODE == mode, (M._FP8_MODE, mode)
    dev = torch.device("cuda:0")
    F8 = torch.float8_e4m3fn
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    res = {"mode": mode, "gpu": torch.cuda.get_device_name(0), "checks": {}, "decode_ms": {}, "write_us": {}}
    log = lambda *a: print(f"[{mode}]", *a, flush=True)

    def call(q, cache, rank, bt, lens, splits, page, out_dtype=torch.bfloat16, rms="log8", sm_scale=1 / 16.0,
             k_scale=None, lse=None):
        B = q.shape[0]
        o = torch.empty(B, HEADS, rank, dtype=out_dtype, device=dev)
        lse = torch.empty(B, HEADS, dtype=torch.float32, device=dev) if lse is None else lse
        lg = torch.empty(B, HEADS, splits, rank + 1, dtype=torch.float32, device=dev)
        fp8 = cache.dtype == F8
        M.decode_attention_fwd(q, cache, cache[..., :rank], o, lse, bt, lens, lg, splits, sm_scale, page,
                               k_scale=k_scale, v_scale=k_scale, is_mla=True, rms_offset=rank + ROPE,
                               rms_bf16=fp8 and rms == "bf16", rms_log8=fp8 and rms == "log8")
        return o, lse

    def width(rank, rms):
        w = rank + ROPE + (2 if rms == "bf16" else 1) * HEADS
        return w + (-w) % 16

    # ---- decode-V and decode-K: bit-exact e4m3 decode inside the kernels
    codes = torch.arange(256, device=dev, dtype=torch.uint8)
    val = codes.view(F8).float()
    ks = torch.tensor(0.0455, device=dev)
    for rank in RANKS:
        page = 16
        # V: one token per row, latent = all codes cycling; every head's output is that row of V
        B = 2
        cache = torch.zeros(B, page, 1, width(rank, "log8"), dtype=torch.uint8, device=dev)
        pos = torch.arange(rank, device=dev)
        cache[:, 0, 0, :rank] = ((pos + 77 * torch.arange(B, device=dev)[:, None]) % 256).to(torch.uint8)
        cache[:, 0, 0, rank:rank + ROPE] = 0x38                           # rope 1.0, rms codes 128 (finite)
        cache[:, 0, 0, rank + ROPE:rank + ROPE + HEADS] = 128
        bt = torch.arange(B, dtype=torch.int32, device=dev)[:, None]
        lens = torch.ones(B, dtype=torch.int32, device=dev)
        q = torch.randn(B, HEADS, rank + ROPE, dtype=torch.bfloat16, device=dev)
        o, _ = call(q, cache.view(F8), rank, bt, lens, 4, page, out_dtype=torch.float32, k_scale=ks)
        c = cache[:, 0, 0, :rank].long()
        want = (val[c] * ks)[:, None, :].expand(B, HEADS, rank)
        fin = torch.isfinite(val[c])[:, None, :].expand(B, HEADS, rank)
        okv = torch.equal(o[fin].view(torch.int32), want[fin].view(torch.int32))
        # K: rms bf16 = 1.0, sm_scale 1, q one-hot -> LSE = the single score = float(code) * k_scale
        L = rank + ROPE
        B = -(-L // HEADS)
        w = width(rank, "bf16")
        cache = torch.zeros(B, page, 1, w, dtype=torch.uint8, device=dev)
        cache[:, 0, 0, :L] = ((torch.arange(L, device=dev) * 7 + 3) % 256).to(torch.uint8)
        cache[:, 0, 0, L:L + 2 * HEADS] = torch.ones(HEADS, dtype=torch.bfloat16, device=dev).view(torch.uint8)
        q = torch.zeros(B, HEADS, L, dtype=torch.bfloat16, device=dev)
        j = torch.arange(B * HEADS, device=dev).clamp(max=L - 1)
        q.view(-1, L)[torch.arange(B * HEADS, device=dev), j] = 1.0
        bt = torch.arange(B, dtype=torch.int32, device=dev)[:, None]
        lens = torch.ones(B, dtype=torch.int32, device=dev)
        _, lse = call(q, cache.view(F8), rank, bt, lens, 4, page, out_dtype=torch.float32, rms="bf16",
                      sm_scale=1.0, k_scale=ks)
        code = cache[0, 0, 0, :L].long()[j]
        want = (val[code] * ks).view(B, HEADS)
        fin = torch.isfinite(val[code]).view(B, HEADS)
        okk = torch.equal(lse[fin].view(torch.int32), want[fin].view(torch.int32))
        res["checks"][f"decode_exact_r{rank}"] = bool(okv and okk)
        log(f"rank {rank}: decode-V bit-exact {okv}, decode-K bit-exact {okk}")

    # ---- attention accuracy vs references
    import test_fp8_kv as T
    T.dev = dev
    for rank in RANKS:
        batch, ctx = 4, 3000
        lat, rp, rms, q, pps = T.make(batch, HEADS, ctx, rank, ROPE, seed=rank)
        scale = torch.tensor(float(max(lat.abs().max(), rp.abs().max())) / 448, device=dev)
        cache = T.fp8_cache(lat, rp, rms, scale, width(rank, "log8"), "log8")
        g = torch.Generator().manual_seed(1)
        bt = torch.randperm(batch * pps, generator=g).reshape(batch, pps).to(torch.int32).to(dev)
        lens = torch.tensor([ctx - (37 * b) % T.PAGE for b in range(batch)], dtype=torch.int32, device=dev)
        o, _ = call(q, cache, rank, bt, lens, 32, T.PAGE, out_dtype=torch.float32, k_scale=scale)
        dl, dr = T.deq(lat, rp, scale)
        r_deq = T.ref(q, dl, dr, T.rms_decoded(rms, scale, "log8"), bt, lens, rank, rms_exact=True)
        r_bf = T.ref(q, lat, rp, rms, bt, lens, rank)
        rel = lambda a, b: ((a.float() - b).abs().max() / b.abs().max()).item()
        e_k, e_q, e_kb = rel(o, r_deq), rel(r_deq, r_bf), rel(o, r_bf)
        ok = e_k < 1e-3 and abs(e_kb - e_q) <= 0.1 * e_q + 1e-3
        res["checks"][f"attention_r{rank}"] = bool(ok)
        res.setdefault("attention_err", {})[str(rank)] = {"vs_deq_ref": e_k, "vs_bf16_ref": e_kb, "quant_err": e_q}
        log(f"rank {rank}: attention vs dequantized ref {e_k:.2e}, vs bf16-KV ref {e_kb:.2e} (quantization {e_q:.2e}) "
            f"{'ok' if ok else 'FAIL'}")
        del lat, rp, rms, cache

    # ---- timing helpers
    flush_buf = torch.empty(48 << 20, dtype=torch.uint8, device=dev)      # 48 MB >> 6 MB L2

    def graph_ms(fn, reps=20):
        fn(); torch.cuda.synchronize()
        s = torch.cuda.Stream()
        with torch.cuda.stream(s):
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, stream=s):
                flush_buf.zero_(); fn()
            gf = torch.cuda.CUDAGraph()
            with torch.cuda.graph(gf, stream=s):
                flush_buf.zero_()
        torch.cuda.synchronize()
        out = []
        for gg in (g, gf):
            for _ in range(3):
                gg.replay()
            ts = []
            for _ in range(reps):
                a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                a.record(); gg.replay(); b.record(); b.synchronize(); ts.append(a.elapsed_time(b))
            out.append(sorted(ts)[len(ts) // 2])
        del g, gf
        return max(out[0] - out[1], 0.0)

    # ---- decode attention timings
    kinds = (["bf16"] if do_bf16 else []) + ["fp8"]
    for rank in RANKS:
        for kind in kinds:
            if time.time() > deadline:
                log("budget exhausted; skipping remaining timings"); res["truncated"] = True; break
            page = PAGE_FP8[rank] if kind == "fp8" else PAGE_BF16[rank]
            w = width(rank, "log8") if kind == "fp8" else rank + 256
            pps = -(-max(CTXS) // page)
            nseq = 3
            dt = torch.uint8 if kind == "fp8" else torch.bfloat16
            cache = torch.empty(nseq * pps, page, 1, w, dtype=dt, device=dev)
            if kind == "fp8":
                cache.copy_((torch.randn(cache.shape, device=dev, dtype=torch.bfloat16) * 40).clamp(-448, 448)
                            .to(F8).view(torch.uint8))
                cache[..., rank + ROPE:rank + ROPE + HEADS] = 128
                cache = cache.view(F8)
            else:
                cache.normal_()
                cache[..., rank + ROPE:rank + ROPE + HEADS] = 1.0
            perm = torch.randperm(nseq * pps, generator=torch.Generator().manual_seed(0)).to(torch.int32).to(dev)
            for c in (1, 3):
                for ctx in CTXS:
                    B = c * (K_MTP + 1)
                    seq = torch.arange(c, device=dev).repeat_interleave(K_MTP + 1)
                    bt = perm.view(nseq, pps)[seq].contiguous()
                    lens = (ctx - K_MTP + torch.arange(K_MTP + 1, device=dev).repeat(c)).to(torch.int32)
                    splits = splitk_cap(B, HEADS, rank + 1, _compute_num_kv_splits(ctx, sms))
                    q = torch.randn(B, HEADS, rank + ROPE, dtype=torch.bfloat16, device=dev) * 0.05
                    o = torch.empty(B, HEADS, rank, dtype=torch.bfloat16, device=dev)
                    lse = torch.empty(B, HEADS, dtype=torch.float32, device=dev)
                    lg = torch.empty(B, HEADS, splits, rank + 1, dtype=torch.float32, device=dev)
                    fp8 = kind == "fp8"
                    fn = lambda: M.decode_attention_fwd(q, cache, cache[..., :rank], o, lse, bt, lens, lg, splits,
                                                        1 / 16.0, page, k_scale=ks, v_scale=ks, is_mla=True,
                                                        rms_offset=rank + ROPE, rms_log8=fp8)
                    ms = graph_ms(fn)
                    gbs = c * ctx * w * (1 if fp8 else 2) / ms / 1e6
                    res["decode_ms"][f"{kind}|{rank}|{c}|{ctx}"] = ms
                    log(f"{kind:4s} rank {rank:4d} c={c} ctx {ctx:6d} splits {splits:3d}: {ms:7.3f} ms  "
                        f"({gbs:5.0f} GB/s of distinct KV)")
            del cache
            torch.cuda.empty_cache()

    # ---- cache write: bytes and time
    if do_bf16:
        import qwen_mla.mla_impl as MI
        import vllm._custom_ops as ops
        impl = MI.QwenMLATritonImpl.__new__(MI.QwenMLATritonImpl)
        rank = 768
        impl.rms_offset, impl.mla_packed_rope, impl.mla_local_heads = rank + ROPE, ROPE, HEADS
        for rms_mode in ("log8", "bf16"):
            os.environ["QWEN_MLA_FP8_RMS"] = rms_mode
            T_ = 4096
            kv_c = torch.randn(T_, rank, device=dev).bfloat16() * 3
            k_pe = torch.cat([torch.randn(T_, 1, ROPE, device=dev) * 2, torch.exp(torch.randn(T_, 1, HEADS, device=dev)),
                              torch.zeros(T_, 1, 256 - ROPE - HEADS, device=dev)], -1).bfloat16()
            slots = torch.randperm(8 * 1664, device=dev)[:T_]
            slots[::97] = -1
            outs = {}
            for path in ("torch", "triton"):
                os.environ["QWEN_MLA_FP8_WRITE"] = path
                cache = torch.zeros(8, 1664, width(rank, rms_mode), dtype=torch.uint8, device=dev).view(F8)
                impl.do_kv_cache_update(kv_c, k_pe, cache, slots, "fp8", ks)
                outs[path] = cache.view(torch.uint8).view(-1, cache.shape[-1])[slots[slots >= 0].long()]
            ok = torch.equal(outs["torch"], outs["triton"])
            res["checks"][f"write_bytes_{rms_mode}"] = bool(ok)
            log(f"cache write ({rms_mode} rms): triton bytes == torch bytes: {ok}")
        os.environ["QWEN_MLA_FP8_RMS"] = "log8"
        for T_ in (4, 12, 8192):
            kv_c = torch.randn(T_, rank, device=dev).bfloat16()
            k_pe = torch.cat([torch.randn(T_, 1, ROPE, device=dev), torch.exp(torch.randn(T_, 1, HEADS, device=dev)),
                              torch.zeros(T_, 1, 256 - ROPE - HEADS, device=dev)], -1).bfloat16()
            slots = torch.randperm(64 * 1664, device=dev)[:T_]
            c8 = torch.zeros(64, 1664, 1024, dtype=F8, device=dev)
            cb = torch.zeros(64, 832, 1024, dtype=torch.bfloat16, device=dev)
            sb = torch.randperm(64 * 832, device=dev)[:T_]
            times = {}
            for path in ("torch", "triton"):
                os.environ["QWEN_MLA_FP8_WRITE"] = path
                times[f"fp8_{path}"] = graph_ms(lambda: impl.do_kv_cache_update(kv_c, k_pe, c8, slots, "fp8", ks)) * 1e3
            times["bf16_concat_and_cache_mla"] = graph_ms(
                lambda: ops.concat_and_cache_mla(kv_c, k_pe.squeeze(1), cb, sb, kv_cache_dtype="auto", scale=ks)) * 1e3
            os.environ["QWEN_MLA_FP8_WRITE"] = "triton"
            res["write_us"][str(T_)] = times
            log(f"cache write, {T_:5d} tokens, one layer: " + ", ".join(f"{k} {v:.1f} us" for k, v in times.items()))
            del c8, cb
    json.dump(res, open(out_path, "w"), indent=1)


# ----------------------------------------------------------------------------------------------- orchestrator
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--modes", default="asm,hoist,triton")
    ap.add_argument("--budget-s", type=float, default=840.0)
    ap.add_argument("--worker", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--deadline", type=float, default=0, help=argparse.SUPPRESS)
    a = ap.parse_args()
    if a.worker:
        mode, do_bf16 = a.worker.split(":")
        return worker(mode, do_bf16 == "1", os.path.join(a.out, f"fp8_bench_{mode}.json"), a.deadline)

    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    deadline = t0 + a.budget_s
    results, errors = {}, []
    modes = a.modes.split(",")
    for i, mode in enumerate(modes):
        left = deadline - time.time()
        if left < 60:
            errors.append(f"{mode}: skipped, budget exhausted"); continue
        env = dict(os.environ, QWEN_MLA_FP8_DECODE=mode)
        # later modes get a share of what is left so the default mode's numbers always come first
        stop = time.time() + left / (len(modes) - i)
        try:
            p = subprocess.run([sys.executable, __file__, "--out", a.out, "--worker", f"{mode}:{int(i == 0)}",
                                "--deadline", str(stop - 30)], env=env, timeout=max(stop - time.time(), 60) + 120)
            if p.returncode:
                errors.append(f"{mode}: worker exit {p.returncode}")
        except subprocess.TimeoutExpired:
            errors.append(f"{mode}: worker timed out")
        f = os.path.join(a.out, f"fp8_bench_{mode}.json")
        if os.path.exists(f):
            results[mode] = json.load(open(f))

    lines, ok = [], True
    base = results.get(modes[0], {})
    for mode, r in results.items():
        for k, v in r["checks"].items():
            ok &= v
            if not v:
                lines.append(f"FAIL check {mode}: {k}")
    if modes[0] not in results:
        ok = False; lines.append(f"FAIL no results for the default mode {modes[0]}")
    dm = base.get("decode_ms", {})
    step = lambda r, kind, c, ctx: sum(n * r["decode_ms"][f"{kind}|{rk}|{c}|{ctx}"] for rk, n in RANKS.items()) \
        if all(f"{kind}|{rk}|{c}|{ctx}" in r["decode_ms"] for rk in RANKS) else None
    lines.append("decode attention per step (16 MLA layers), ms -- bf16 cache vs FP8 per mode (ratio to bf16):")
    for c in (1, 3):
        for ctx in CTXS:
            b = step(base, "bf16", c, ctx) if base else None
            cells = [f"c={c} ctx {ctx:6d}: bf16 {b:7.3f}" if b else f"c={c} ctx {ctx:6d}: bf16    n/a"]
            for mode, r in results.items():
                f = step(r, "fp8", c, ctx)
                cells.append(f"{mode} {f:7.3f} ({f / b:.2f}x)" if f and b else f"{mode} n/a")
                if mode == modes[0] and ctx >= 65536 and not (f and b and f <= 1.02 * b):
                    ok = False
            lines.append("  " + " | ".join(cells))
    lines.append("per layer type, default mode, FP8/bf16 time ratio (c=1 / c=3):")
    for rk in RANKS:
        cells = []
        for ctx in CTXS:
            rs = []
            for c in (1, 3):
                b, f = dm.get(f"bf16|{rk}|{c}|{ctx}"), dm.get(f"fp8|{rk}|{c}|{ctx}")
                rs.append(f"{f / b:.2f}" if b and f else "n/a")
            cells.append(f"{ctx // 1024}k {'/'.join(rs)}")
        lines.append(f"  rank {rk:4d}: " + "  ".join(cells))
    wu = base.get("write_us", {})
    for n, t in wu.items():
        lines.append(f"cache write {int(n):5d} tokens/layer: " + ", ".join(f"{k} {v:.1f} us" for k, v in t.items()))
        if int(n) <= 12 and t["fp8_triton"] > t["fp8_torch"]:
            ok = False; lines.append(f"FAIL: Triton write slower than torch at {n} tokens")
    if not wu:
        ok = False; lines.append("FAIL: no cache-write timings")
    best = {}
    for mode, r in results.items():
        tot = [step(r, "fp8", c, ctx) for c in (1, 3) for ctx in CTXS if ctx >= 65536]
        if all(tot):
            best[mode] = sum(tot)
    if best:
        lines.append("sum of per-step FP8 times at >= 64k, by mode: " +
                     ", ".join(f"{m} {v:.2f} ms" for m, v in sorted(best.items(), key=lambda kv: kv[1])))
    lines += errors
    if errors and modes[0] in [e.split(":")[0] for e in errors]:
        ok = False
    verdict = "PASS" if ok else "FAIL"
    lines.append(f"{verdict}  ({time.time() - t0:.0f} s; GPU {base.get('gpu', '?')})")
    txt = "\n".join(lines)
    print(txt, flush=True)
    open(os.path.join(a.out, "fp8_bench.txt"), "w").write(txt + "\n")
    json.dump({"verdict": verdict, "results": results, "errors": errors, "summary": lines},
              open(os.path.join(a.out, "fp8_bench.json"), "w"), indent=1)
    return 0 if ok else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SystemExit:
        raise
    except Exception as e:                              # noqa: BLE001
        import traceback; traceback.print_exc()
        sys.exit(2)
