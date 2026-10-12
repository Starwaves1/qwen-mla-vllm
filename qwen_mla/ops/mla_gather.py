"""Torch fallback for gather_and_maybe_dequant_cache at non-DeepSeek cache widths.

    RuntimeError: gather_and_maybe_dequant_cache only support the head_dim to
                  320 or 576 for better performance

The CUDA kernel hard-rejects any entry size outside DeepSeek's two. Our rows are
kv_lora_rank + tail (524 / 1048 / 2096 at TP=2), so every chunked-context prefill dies -- which
is every request long enough to matter, and none of the short ones we validated on.

SEMANTICS, read off the metadata builder rather than guessed (mla_attention.py ~1646):
    chunk_starts[chunk][seq]  -> seq_starts : where this chunk begins IN THE SEQUENCE
    cu_seq_lens[chunk]        -> [0, cumsum(chunk_seq_lens)] : where each seq begins IN dst
    token_to_seq[chunk]       -> repeat_interleave(seq_idx, chunk_seq_lens)
so for dst row j:  s = token_to_seq[j];  tok = seq_starts[s] + (j - cu_seq_lens[s])
and tok indexes the sequence, which the block table maps into the paged cache.

test_gather_torch.py checks this against the CUDA op itself at width 576, so the reading is
verified rather than asserted -- a plausible-but-wrong reading here would degrade long-context
attention while leaving the output fluent.
"""
from __future__ import annotations

import torch

_DEQUANT_SLICE = 8192   # rows per FP8 dequant slice (see the FP8 branch)


def gather_cache_torch(*, src_cache: torch.Tensor, dst: torch.Tensor,
                       block_table: torch.Tensor, cu_seq_lens: torch.Tensor,
                       token_to_seq: torch.Tensor, num_tokens: int,
                       kv_cache_dtype: str, scale: torch.Tensor,
                       seq_starts: torch.Tensor | None = None, mla_rms=None) -> None:
    fp8 = str(kv_cache_dtype).startswith("fp8")
    if not fp8 and kv_cache_dtype not in ("auto", "bfloat16", "float16"):
        raise NotImplementedError(f"MLA gather fallback has no dequant path for kv_cache_dtype={kv_cache_dtype}")
    if fp8 and mla_rms is None:
        raise NotImplementedError("FP8 MLA gather needs the row layout (rms offset, heads)")
    if num_tokens <= 0:
        return

    block_size = src_cache.shape[1]
    j = torch.arange(num_tokens, device=dst.device)
    s = token_to_seq[:num_tokens].long()
    tok = j - cu_seq_lens.long()[s]
    if seq_starts is not None:
        tok = tok + seq_starts.long()[s]
    blk = block_table[s, tok // block_size].long()
    if not fp8:
        dst[:num_tokens].copy_(src_cache[blk, tok % block_size].reshape(num_tokens, -1))
        return
    # FP8 row: [latent | rope] e4m3 (x scale) then the rms tail as raw bf16 (2 bytes per head). The
    # destination is the bf16 workspace the prefill path splits on the BF16 layout, where rms sits at
    # rms_offset as one element per head -- so decode into exactly that.
    #
    # Dequantized in slices: the fp32 intermediates cost ~10 bytes per latent element, so one pass over
    # a 64k-row chunk is several hundred MB of transient on top of the workspace -- unaccounted by vLLM's
    # profile run, it is what ran 16 GB cards out of memory mid-prefill. Slicing keeps results identical.
    off, nh = mla_rms[:2]
    mode = mla_rms[2] if len(mla_rms) > 2 else "bf16"
    u8 = src_cache.view(torch.uint8)
    for a in range(0, num_tokens, _DEQUANT_SLICE):
        b = min(a + _DEQUANT_SLICE, num_tokens)
        bk, tk = blk[a:b], tok[a:b] % block_size
        rows = u8[bk, tk].reshape(b - a, -1)
        out = dst[a:b]
        if mode == "e4m3":                 # compact: the whole row (rms included) is e4m3 x scale
            out[:, :off + nh].copy_((rows[:, :off + nh].view(torch.float8_e4m3fn).float() * scale.float()).to(dst.dtype))
            continue
        out[:, :off].copy_((rows[:, :off].view(torch.float8_e4m3fn).float() * scale.float()).to(dst.dtype))
        if mode == "log8":                 # rms as a uint8 code of log2(rms)
            from ..mla_impl import RMS_LOG2_HI, RMS_LOG2_LO
            code = rows[:, off:off + nh].float()
            out[:, off:off + nh].copy_(
                torch.exp2(RMS_LOG2_LO + code * ((RMS_LOG2_HI - RMS_LOG2_LO) / 255.0)).to(dst.dtype))
        else:
            out[:, off:off + nh].copy_(rows[:, off:off + 2 * nh].contiguous().view(torch.bfloat16).to(dst.dtype))
