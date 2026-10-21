// Experimental single-pass INT8 MLA decode for the "int8-tok" cache on sm_86 (RTX 3090). Not used by prod.
// Same cache layout, q prep and split/LSE outputs as qwen_mla/ops/mla_int8_decode.py (_int8_decode_kernel);
// the Triton _int8_reduce_kernel combines the splits.
//
// CTA = (12-head block, kv split, request): 48 query rows = 3 m16 tiles (12 heads x 4 MTP rows), 12 warps.
// Warp w = (m = w / 4, c = w % 4): m16 row tile m, latent column slice c (R / 4 columns).
//   QK : q fragments of (m, c) live in registers; S partial [16, BN] over the slice via int8 mma m16n8k32,
//        K fragments by ldmatrix from a 2-stage cp.async ring; rope (own KV group, 64 dims) adds 1 k-step on
//        8 of the 12 warps. Partials are scaled to fp32 and exchanged through shared memory.
//   softmax: every warp of row tile m sums the 4 partials in the same order (bit-identical S), so the online
//        softmax state is replicated across the 4 warps and P never leaves registers.
//   PV : P (with sigma_t folded in) quantized to int8 per (row, tile) feeds the A operand straight from the S
//        accumulator layout; V (token-major int8) comes from ldmatrix.trans (b16 view) + 2 PRMT per register,
//        so the key order of A and B match without a byte gather. int32 per tile, fp32 rescale.
// Two barriers per key tile (tile landed / S partials ready).
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>

#ifndef EXP
#define EXP 0   // experiment switches for ablations (0 = normal build)
#endif

namespace {

constexpr int NWARP = 12, NTHR = NWARP * 32;
constexpr int MAGIC_I = 0x4B400000;          // int32 x + MAGIC_I reinterpreted as fp32 = 1.5 * 2^23 + x (|x| < 2^22)
constexpr float MAGIC_F = 12582912.0f;

struct Params {
  const int8_t* q8; const float* qs; const int8_t* qr8; const float* qrs;
  const uint8_t* cache; const int* bt; const int* lens;
  float* opart; float* lsepart; float* sdbg;
  long long page_bytes; int stride_bt; int nsplit; int page; int ld_dbg;
};

__device__ __forceinline__ uint32_t smem_u32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void cp_async16(uint32_t dst, const void* src) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(dst), "l"(src));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
__device__ __forceinline__ void cp_wait_all() { asm volatile("cp.async.wait_group 0;\n" ::: "memory"); }
__device__ __forceinline__ void ldsm4(uint32_t a, uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(a) : "memory");
}
__device__ __forceinline__ void ldsm4t(uint32_t a, uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(a) : "memory");
}
__device__ __forceinline__ void mma_s8(int (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ float i2f(int x) { return __int_as_float(x) - MAGIC_F; }
__device__ __forceinline__ uint32_t ld32(const int8_t* p) { return *reinterpret_cast<const uint32_t*>(p); }

template <int R, int BN>
struct Cfg {
  static constexpr int LSTR = R + 16, RSTR = 144;          // padded rows: ldmatrix of 8 rows is conflict-free
  static constexpr int ROPE = BN * LSTR;
  static constexpr int SIG = ROPE + BN * RSTR;
  static constexpr int RHO = SIG + ((BN * 2 + 15) / 16) * 16;
  static constexpr int RMS = RHO + ((BN * 2 + 15) / 16) * 16;
  static constexpr int STAGE = ((RMS + 12 * BN) + 127) / 128 * 128;
  static constexpr int PART = 2 * STAGE;
  static constexpr int PART_B = 12 * (BN / 8) * 32 * 16;
  static constexpr int IVS = BN + 8;                         // inv-rms row stride (floats)
  static constexpr int INVR = PART + PART_B;
  static constexpr int RMX = INVR + 12 * IVS * 4;          // float2 [3 m][16 rows][4 c] (row max, max log2 p*sigma)
  static constexpr int PFR = RMX + 3 * 16 * 4 * 8;          // uint32 [3 m][BN / 32][32 lanes][4]: P int8 A fragments
  static constexpr int LSUM = PFR + 3 * (BN / 32) * 32 * 16; // float [48 rows][4 c]
  static constexpr int SMEM = LSUM + 48 * 4 * 4;
};

template <int R, int BN, int MODE>   // MODE 0: QK + softmax-free score pass (debug / pipeline probe), 1: full
__global__ void __launch_bounds__(NTHR, 1) int8_decode_kernel(const Params p) {
  using C = Cfg<R, BN>;
  constexpr int NT = BN / 8, KB = BN / 32, CW = R / 4, KS = CW / 32, CH = CW / 16;
  static_assert(KS * 32 * 127 * 127 < (1 << 22), "QK partial exceeds the magic-number range");
  extern __shared__ __align__(128) uint8_t smem[];
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
  const int m = warp >> 2, c = warp & 3;
  const int hb = blockIdx.x, split = blockIdx.y, req = blockIdx.z;
  const int L = p.lens[req];
  const int per = ((L + p.nsplit - 1) / p.nsplit + BN - 1) / BN * BN;
  const int start = split * per;
  const int end = min(start + per, L);
  const int ntile = end > start ? (end - start + BN - 1) / BN : 0;
  const int qrow0 = (req * 24 + hb * 12) * 4 + 16 * m;

  uint32_t qa[KS][4];
  {
    const int8_t* qg = p.q8 + (size_t)(qrow0 + g) * R + c * CW + 4 * t;
    const int8_t* qh = qg + 8 * R;
#pragma unroll
    for (int ks = 0; ks < KS; ++ks) {
      qa[ks][0] = ld32(qg + ks * 32); qa[ks][1] = ld32(qh + ks * 32);
      qa[ks][2] = ld32(qg + ks * 32 + 16); qa[ks][3] = ld32(qh + ks * 32 + 16);
    }
  }
  // rope k-steps: row tile 0 = heads 0-3 (local KV group 0), 1 = heads 4-7 (groups 0, 1), 2 = heads 8-11 (group 1)
  int rlg = -1, rks = 0;
  if (m == 0 && c < 2) { rlg = 0; rks = c; }
  else if (m == 1) { rlg = c >> 1; rks = c & 1; }
  else if (m == 2 && c >= 2) { rlg = 1; rks = c - 2; }
  const bool has_rope = rlg >= 0;
  uint32_t qra[4] = {0u, 0u, 0u, 0u};
  if (has_rope) {
    const int8_t* rg = p.qr8 + (size_t)(qrow0 + g) * 256 + (2 * hb + rlg) * 64 + rks * 32 + 4 * t;
    qra[0] = ld32(rg); qra[1] = ld32(rg + 8 * 256); qra[2] = ld32(rg + 16); qra[3] = ld32(rg + 8 * 256 + 16);
  }
  const float qs_g = p.qs[qrow0 + g], qs_h = p.qs[qrow0 + g + 8];
  const float qrs_g = p.qrs[qrow0 + g], qrs_h = p.qrs[qrow0 + g + 8];
  const int lim = min(end, L - 3 + (g & 3));     // rows g and g + 8 share the MTP index j = g % 4
  const int hl_g = 4 * m + (g >> 2), hl_h = hl_g + 2;

  const int* btr = p.bt + (size_t)req * p.stride_bt;
  // cp.async work split: every thread copies LPT latent chunks (rows r0 + k * RSTEP, column chunk lch) and at most
  // one rope / scale / rms chunk; only the page base and the token-in-page offset change per tile.
  constexpr int LCH = R / 16, RSTEP = NTHR / LCH;
  constexpr bool LFAST = NTHR % LCH == 0 && BN % RSTEP == 0;   // rows r0 + k * RSTEP, fixed column chunk
  constexpr int LPT = LFAST ? BN / RSTEP : (BN * LCH + NTHR - 1) / NTHR;
  static_assert(BN * 8 + BN / 4 + 12 * (BN / 16) <= NTHR, "one small chunk per thread");
  const int lr0 = tid / LCH, lch = tid - (tid / LCH) * LCH;
  const uint32_t l_soff = lr0 * C::LSTR + lch * 16;
  const uint32_t l_goff = lr0 * R + lch * 16;
  // small chunk: rope rows (tid < BN*8), then sigma, rho, rms
  uint32_t x_soff = 0; long long x_goff = 0; int x_tokmul = 0; bool x_has = true;
  if (tid < BN * 8) {
    const int r = tid >> 3, ch = tid & 7;
    x_soff = C::ROPE + r * C::RSTR + ch * 16; x_goff = (long long)p.page * R + r * 256 + hb * 128 + ch * 16; x_tokmul = 256;
  } else if (tid < BN * 8 + BN / 8) {
    const int ch = tid - BN * 8;
    x_soff = C::SIG + ch * 16; x_goff = (long long)p.page * (R + 256) + ch * 16; x_tokmul = 2;
  } else if (tid < BN * 8 + BN / 4) {
    const int ch = tid - BN * 8 - BN / 8;
    x_soff = C::RHO + ch * 16; x_goff = (long long)p.page * (R + 258) + ch * 16; x_tokmul = 2;
  } else if (tid < BN * 8 + BN / 4 + 12 * (BN / 16)) {
    const int i = tid - BN * 8 - BN / 4, hl = i / (BN / 16), ch = i % (BN / 16);
    x_soff = C::RMS + hl * BN + ch * 16; x_goff = (long long)p.page * (R + 260) + (long long)(12 * hb + hl) * p.page + ch * 16; x_tokmul = 1;
  } else {
    x_has = false;
  }
  const uint32_t smem0 = smem_u32(smem);
  auto issue = [&](int it, int stg, int pg) {
    const int tip = (start + it * BN) % p.page;
    const uint8_t* pb = p.cache + (size_t)pg * p.page_bytes;
    const uint32_t sb = smem0 + stg * C::STAGE;
    const uint8_t* lg = pb + (size_t)tip * R + l_goff;
    if constexpr (LFAST) {
#pragma unroll
      for (int k = 0; k < LPT; ++k) if (EXP != 4) cp_async16(sb + l_soff + k * RSTEP * C::LSTR, lg + (size_t)k * RSTEP * R);
    } else {
#pragma unroll
      for (int k = 0; k < LPT; ++k) {
        const int i = tid + k * NTHR, r = i / LCH, ch = i % LCH;
        if (i < BN * LCH) cp_async16(sb + r * C::LSTR + ch * 16, pb + (size_t)(tip + r) * R + ch * 16);
      }
    }
    if (x_has && EXP != 4) cp_async16(sb + x_soff, pb + x_goff + (size_t)tip * x_tokmul);
    cp_commit();
  };

  float m_g = -INFINITY, m_h = -INFINITY, l_g = 0.f, l_h = 0.f;
  float acc[MODE ? CH : 1][8];
#pragma unroll
  for (int ch = 0; ch < (MODE ? CH : 1); ++ch)
#pragma unroll
    for (int e = 0; e < 8; ++e) acc[ch][e] = 0.f;
  float4* part = reinterpret_cast<float4*>(smem + C::PART);
  float* invr = reinterpret_cast<float*>(smem + C::INVR);
  float2* rmx = reinterpret_cast<float2*>(smem + C::RMX);
  uint32_t* pfr = reinterpret_cast<uint32_t*>(smem + C::PFR);
  float* lsum = reinterpret_cast<float*>(smem + C::LSUM);

#ifdef PROF
  long long pt[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  long long tc = clock64();
#define TICK(i) { long long tn = clock64(); pt[i] += tn - tc; tc = tn; }
#else
#define TICK(i)
#endif
  int pg_next = 0;
  if (ntile > 0) {
    issue(0, 0, btr[start / p.page]);
    if (ntile > 1) pg_next = btr[(start + BN) / p.page];
  }
  for (int it = 0; it < ntile; ++it) {
    TICK(7)
    cp_wait_all();
    __syncthreads();                                         // B1: tile it landed; tile it-1 fully consumed
    TICK(0)
    if (it + 1 < ntile) {
      issue(it + 1, (it + 1) & 1, pg_next);
      if (it + 2 < ntile) pg_next = btr[(start + (it + 2) * BN) / p.page];
    }
    TICK(1)
    const uint8_t* st = smem + (it & 1) * C::STAGE;
    const uint32_t sb = smem_u32(st);
    const int tok0 = start + it * BN;
    for (int i = tid; i < 12 * BN; i += NTHR) {
      const int hl = i / BN, k = i % BN;
      invr[hl * C::IVS + k] = exp2f(3.f - (float)st[C::RMS + hl * BN + k] * (6.f / 255.f));
    }

    if constexpr (MODE == 2) { __syncthreads(); continue; }
    // ---- QK partial over this warp's column slice (+ rope k-step), scaled to fp32
    float own[NT][4];
#pragma unroll
    for (int np = 0; np < NT / 2; ++np) {
      int sa[2][4] = {{MAGIC_I, MAGIC_I, MAGIC_I, MAGIC_I}, {MAGIC_I, MAGIC_I, MAGIC_I, MAGIC_I}};
      const int key = np * 16 + ((lane >> 4) << 3) + (lane & 7);
      const uint32_t la = sb + key * C::LSTR + c * CW + ((lane >> 3) & 1) * 16;
#pragma unroll
      for (int ks = 0; ks < KS; ++ks) {
        uint32_t b0, b1, b2, b3;
        ldsm4(la + ks * 32, b0, b1, b2, b3);
        mma_s8(sa[0], qa[ks], b0, b1);
        mma_s8(sa[1], qa[ks], b2, b3);
      }
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const int nt = 2 * np + h;
        const float2 sg = __half22float2(*reinterpret_cast<const __half2*>(st + C::SIG + (8 * nt + 2 * t) * 2));
        own[nt][0] = i2f(sa[h][0]) * (qs_g * sg.x); own[nt][1] = i2f(sa[h][1]) * (qs_g * sg.y);
        own[nt][2] = i2f(sa[h][2]) * (qs_h * sg.x); own[nt][3] = i2f(sa[h][3]) * (qs_h * sg.y);
      }
      if (has_rope) {
        int ra[2][4] = {{MAGIC_I, MAGIC_I, MAGIC_I, MAGIC_I}, {MAGIC_I, MAGIC_I, MAGIC_I, MAGIC_I}};
        uint32_t b0, b1, b2, b3;
        ldsm4(sb + C::ROPE + key * C::RSTR + rlg * 64 + rks * 32 + ((lane >> 3) & 1) * 16, b0, b1, b2, b3);
        mma_s8(ra[0], qra, b0, b1);
        mma_s8(ra[1], qra, b2, b3);
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          const int nt = 2 * np + h;
          const float2 rh = __half22float2(*reinterpret_cast<const __half2*>(st + C::RHO + (8 * nt + 2 * t) * 2));
          own[nt][0] = fmaf(i2f(ra[h][0]), qrs_g * rh.x, own[nt][0]);
          own[nt][1] = fmaf(i2f(ra[h][1]), qrs_g * rh.y, own[nt][1]);
          own[nt][2] = fmaf(i2f(ra[h][2]), qrs_h * rh.x, own[nt][2]);
          own[nt][3] = fmaf(i2f(ra[h][3]), qrs_h * rh.y, own[nt][3]);
        }
      }
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const int nt = 2 * np + h;
        part[((m * 4 + c) * NT + nt) * 32 + lane] = make_float4(own[nt][0], own[nt][1], own[nt][2], own[nt][3]);
      }
    }
    TICK(2)
    __syncthreads();                                         // B2: partials and inv-rms ready
    TICK(3)

    // ---- S for (row tile m, n-tiles c, c + 4, ..): the 4 partials summed in a fixed order, / rms, masked
    constexpr int NJ = NT / 4;
    float s[NJ][4], sgm[NJ][2];
    float mx_g = -INFINITY, mx_h = -INFINITY, ux_g = -INFINITY, ux_h = -INFINITY;
#pragma unroll
    for (int j = 0; j < NJ; ++j) {
      const int nt = c + 4 * j;
      float4 a = part[((m * 4 + 0) * NT + nt) * 32 + lane];
#pragma unroll
      for (int cc = 1; cc < 4; ++cc) {
        const float4 b = part[((m * 4 + cc) * NT + nt) * 32 + lane];
        a.x += b.x; a.y += b.y; a.z += b.z; a.w += b.w;
      }
      const int kp = tok0 + 8 * nt + 2 * t;
      const bool v0 = kp < lim, v1 = kp + 1 < lim;
      const float2 ig = *reinterpret_cast<const float2*>(invr + hl_g * C::IVS + 8 * nt + 2 * t);
      const float2 ih = *reinterpret_cast<const float2*>(invr + hl_h * C::IVS + 8 * nt + 2 * t);
      const float2 sg = __half22float2(*reinterpret_cast<const __half2*>(st + C::SIG + (8 * nt + 2 * t) * 2));
      s[j][0] = v0 ? a.x * ig.x : -INFINITY;
      s[j][1] = v1 ? a.y * ig.y : -INFINITY;
      s[j][2] = v0 ? a.z * ih.x : -INFINITY;
      s[j][3] = v1 ? a.w * ih.y : -INFINITY;
      sgm[j][0] = v0 ? sg.x : 0.f;
      sgm[j][1] = v1 ? sg.y : 0.f;
      const float l0 = v0 ? __log2f(sg.x) : -INFINITY, l1 = v1 ? __log2f(sg.y) : -INFINITY;
      mx_g = fmaxf(mx_g, fmaxf(s[j][0], s[j][1]));
      mx_h = fmaxf(mx_h, fmaxf(s[j][2], s[j][3]));
      ux_g = fmaxf(ux_g, fmaxf(s[j][0] + l0, s[j][1] + l1));     // max of log2(p * sigma) + m: exact P' scale
      ux_h = fmaxf(ux_h, fmaxf(s[j][2] + l0, s[j][3] + l1));
    }
    if constexpr (MODE != 1) {
      m_g = fmaxf(m_g, mx_g); m_h = fmaxf(m_h, mx_h);
      if (p.sdbg != nullptr) {
#pragma unroll
        for (int j = 0; j < NJ; ++j) {
          const int kp = tok0 + 8 * (c + 4 * j) + 2 * t;
          float* dg = p.sdbg + (size_t)(qrow0 + g) * p.ld_dbg;
          float* dh = p.sdbg + (size_t)(qrow0 + g + 8) * p.ld_dbg;
          if (kp < end) { dg[kp] = s[j][0]; dh[kp] = s[j][2]; }
          if (kp + 1 < end) { dg[kp + 1] = s[j][1]; dh[kp + 1] = s[j][3]; }
        }
      }
      continue;
    }
#pragma unroll
    for (int o = 1; o <= 2; o <<= 1) {
      mx_g = fmaxf(mx_g, __shfl_xor_sync(0xffffffffu, mx_g, o));
      mx_h = fmaxf(mx_h, __shfl_xor_sync(0xffffffffu, mx_h, o));
      ux_g = fmaxf(ux_g, __shfl_xor_sync(0xffffffffu, ux_g, o));
      ux_h = fmaxf(ux_h, __shfl_xor_sync(0xffffffffu, ux_h, o));
    }
    if (t == 0) {
      rmx[(m * 16 + g) * 4 + c] = make_float2(mx_g, ux_g);
      rmx[(m * 16 + g + 8) * 4 + c] = make_float2(mx_h, ux_h);
    }
    TICK(5)
    __syncthreads();                                         // B3: per-warp row maxima ready
    TICK(6)
    {
      const float4 a0 = *reinterpret_cast<const float4*>(rmx + (m * 16 + g) * 4);
      const float4 a1 = *reinterpret_cast<const float4*>(rmx + (m * 16 + g) * 4 + 2);
      const float4 b0 = *reinterpret_cast<const float4*>(rmx + (m * 16 + g + 8) * 4);
      const float4 b1 = *reinterpret_cast<const float4*>(rmx + (m * 16 + g + 8) * 4 + 2);
      mx_g = fmaxf(fmaxf(a0.x, a0.z), fmaxf(a1.x, a1.z)); ux_g = fmaxf(fmaxf(a0.y, a0.w), fmaxf(a1.y, a1.w));
      mx_h = fmaxf(fmaxf(b0.x, b0.z), fmaxf(b1.x, b1.z)); ux_h = fmaxf(fmaxf(b0.y, b0.w), fmaxf(b1.y, b1.w));
    }
    const float mn_g = fmaxf(m_g, mx_g), mn_h = fmaxf(m_h, mx_h);
    const float ms_g = mn_g == -INFINITY ? 0.f : mn_g, ms_h = mn_h == -INFINITY ? 0.f : mn_h;
    const float al_g = exp2f(m_g - ms_g), al_h = exp2f(m_h - ms_h);
    m_g = mn_g; m_h = mn_h;
    const float rm_g = exp2f(ux_g - ms_g), rm_h = exp2f(ux_h - ms_h);   // max over the tile of p * sigma, per row
    const float sc_g = rm_g > 0.f ? 127.f / rm_g : 0.f, sc_h = rm_h > 0.f ? 127.f / rm_h : 0.f;
    const float rs_g = rm_g * (1.f / 127.f), rs_h = rm_h * (1.f / 127.f);
    float ps_g = 0.f, ps_h = 0.f;
#pragma unroll
    for (int j = 0; j < NJ; ++j) {
      const int nt = c + 4 * j, kb = nt >> 2, ntl = nt & 3;
      const float p0 = exp2f(s[j][0] - ms_g), p1 = exp2f(s[j][1] - ms_g);
      const float p2 = exp2f(s[j][2] - ms_h), p3 = exp2f(s[j][3] - ms_h);
      ps_g += p0 + p1; ps_h += p2 + p3;
      const uint32_t q0 = __float_as_uint(fmaf(p0 * sgm[j][0], sc_g, MAGIC_F)), q1 = __float_as_uint(fmaf(p1 * sgm[j][1], sc_g, MAGIC_F));
      const uint32_t q2 = __float_as_uint(fmaf(p2 * sgm[j][0], sc_h, MAGIC_F)), q3 = __float_as_uint(fmaf(p3 * sgm[j][1], sc_h, MAGIC_F));
      // A fragment of the PV mma for key block kb: regs 0/2 = row g, 1/3 = row g + 8; n-tile ntl -> reg (ntl / 2) * 2, half ntl % 2
      uint16_t* pf = reinterpret_cast<uint16_t*>(pfr + ((m * KB + kb) * 32 + lane) * 4);
      pf[(ntl >> 1) * 4 + (ntl & 1)] = (uint16_t)__byte_perm(q0, q1, 0x0040);
      pf[(ntl >> 1) * 4 + 2 + (ntl & 1)] = (uint16_t)__byte_perm(q2, q3, 0x0040);
    }
    l_g = l_g * al_g + ps_g; l_h = l_h * al_h + ps_h;        // partial row sums (this lane's keys), reduced at the end
    __syncthreads();                                         // B4: P (int8, A-fragment order) ready
    uint32_t pa[KB][4];
#pragma unroll
    for (int kb = 0; kb < KB; ++kb) {
      const uint4 v = *reinterpret_cast<const uint4*>(pfr + ((m * KB + kb) * 32 + lane) * 4);
      pa[kb][0] = v.x; pa[kb][1] = v.y; pa[kb][2] = v.z; pa[kb][3] = v.w;
    }
    {
      TICK(4)
      const bool resc = EXP == 1 ? false : __any_sync(0xffffffffu, (al_g != 1.f) || (al_h != 1.f));
      if constexpr (MODE == 3) { acc[0][0] += __uint_as_float(pa[0][0] ^ pa[0][1] ^ pa[0][2] ^ pa[0][3]) + (resc ? 1.f : 0.f); continue; }
      // ---- PV over this warp's column slice: 16 columns per ldmatrix.x4.trans (even / odd column n-tiles)
#pragma unroll
      for (int ch = 0; ch < (EXP == 2 ? CH / 2 : CH); ++ch) {
        int de[4] = {MAGIC_I, MAGIC_I, MAGIC_I, MAGIC_I}, dd[4] = {MAGIC_I, MAGIC_I, MAGIC_I, MAGIC_I};
#pragma unroll
        for (int kb = 0; kb < KB; ++kb) {
          uint32_t r0, r1, r2, r3;
          ldsm4t(sb + (32 * kb + lane) * C::LSTR + c * CW + ch * 16, r0, r1, r2, r3);
          mma_s8(de, pa[kb], __byte_perm(r0, r1, 0x6420), __byte_perm(r2, r3, 0x6420));
          mma_s8(dd, pa[kb], __byte_perm(r0, r1, 0x7531), __byte_perm(r2, r3, 0x7531));
        }
        if (resc) {
#pragma unroll
          for (int e = 0; e < 4; ++e) { acc[ch][e] *= al_g; acc[ch][4 + e] *= al_h; }
        }
        if (EXP == 3) {
#pragma unroll
          for (int e = 0; e < 4; ++e) { acc[ch][e] = __int_as_float(__float_as_int(acc[ch][e]) + de[e]); acc[ch][4 + e] = __int_as_float(__float_as_int(acc[ch][4 + e]) + dd[e]); }
          continue;
        }
        acc[ch][0] = fmaf(i2f(de[0]), rs_g, acc[ch][0]);
        acc[ch][1] = fmaf(i2f(dd[0]), rs_g, acc[ch][1]);
        acc[ch][2] = fmaf(i2f(de[1]), rs_g, acc[ch][2]);
        acc[ch][3] = fmaf(i2f(dd[1]), rs_g, acc[ch][3]);
        acc[ch][4] = fmaf(i2f(de[2]), rs_h, acc[ch][4]);
        acc[ch][5] = fmaf(i2f(dd[2]), rs_h, acc[ch][5]);
        acc[ch][6] = fmaf(i2f(de[3]), rs_h, acc[ch][6]);
        acc[ch][7] = fmaf(i2f(dd[3]), rs_h, acc[ch][7]);
      }
    }
  }

#ifdef PROF
  TICK(7)
  if (lane == 0 && p.sdbg != nullptr) {
    float* d = p.sdbg + (((size_t)blockIdx.z * gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x) * 12 * 8 + warp * 8;
    for (int i = 0; i < 8; ++i) d[i] = (float)pt[i] / (ntile > 0 ? ntile : 1);
  }
#endif
  if constexpr (MODE == 0 || MODE == 2) {
    if (c == 0 && t == 0) {
      p.lsepart[(qrow0 + g) * p.nsplit + split] = m_g;
      p.lsepart[(qrow0 + g + 8) * p.nsplit + split] = m_h;
    }
  } else {
    l_g += __shfl_xor_sync(0xffffffffu, l_g, 1); l_h += __shfl_xor_sync(0xffffffffu, l_h, 1);
    l_g += __shfl_xor_sync(0xffffffffu, l_g, 2); l_h += __shfl_xor_sync(0xffffffffu, l_h, 2);
    if (t == 0) { lsum[(m * 16 + g) * 4 + c] = l_g; lsum[(m * 16 + g + 8) * 4 + c] = l_h; }
    __syncthreads();
    {
      const float4 a = *reinterpret_cast<const float4*>(lsum + (m * 16 + g) * 4);
      const float4 b = *reinterpret_cast<const float4*>(lsum + (m * 16 + g + 8) * 4);
      l_g = ((a.x + a.y) + a.z) + a.w; l_h = ((b.x + b.y) + b.z) + b.w;
    }
    const float ig = l_g > 0.f ? 1.f / l_g : 0.f, ih = l_h > 0.f ? 1.f / l_h : 0.f;
    float* og = p.opart + ((size_t)(qrow0 + g) * p.nsplit + split) * R + c * CW + 4 * t;
    float* oh = p.opart + ((size_t)(qrow0 + g + 8) * p.nsplit + split) * R + c * CW + 4 * t;
#pragma unroll
    for (int ch = 0; ch < CH; ++ch) {
      *reinterpret_cast<float4*>(og + ch * 16) = make_float4(acc[ch][0] * ig, acc[ch][1] * ig, acc[ch][2] * ig, acc[ch][3] * ig);
      *reinterpret_cast<float4*>(oh + ch * 16) = make_float4(acc[ch][4] * ih, acc[ch][5] * ih, acc[ch][6] * ih, acc[ch][7] * ih);
    }
    if (c == 0 && t == 0) {
      p.lsepart[(qrow0 + g) * p.nsplit + split] = l_g > 0.f ? m_g + log2f(l_g) : -INFINITY;
      p.lsepart[(qrow0 + g + 8) * p.nsplit + split] = l_h > 0.f ? m_h + log2f(l_h) : -INFINITY;
    }
  }
}

template <int R, int BN, int MODE>
int launch(const Params& p, int nreq, cudaStream_t s) {
  using C = Cfg<R, BN>;
  static bool init = false;
  if (!init) {
    cudaError_t e = cudaFuncSetAttribute(int8_decode_kernel<R, BN, MODE>, cudaFuncAttributeMaxDynamicSharedMemorySize, C::SMEM);
    if (e != cudaSuccess) return (int)e;
    init = true;
  }
  int8_decode_kernel<R, BN, MODE><<<dim3(2, p.nsplit, nreq), NTHR, C::SMEM, s>>>(p);
  return (int)cudaGetLastError();
}

}  // namespace

extern "C" int mla_int8_decode(const void* q8, const void* qs, const void* qr8, const void* qrs, const void* cache,
                               const void* bt, const void* lens, void* opart, void* lsepart, void* sdbg, int ld_dbg,
                               long long page_bytes, int stride_bt, int nsplit, int page, int nreq, int rank, int bn,
                               int mode, void* stream) {
  Params p{(const int8_t*)q8, (const float*)qs, (const int8_t*)qr8, (const float*)qrs, (const uint8_t*)cache,
           (const int*)bt, (const int*)lens, (float*)opart, (float*)lsepart, (float*)sdbg,
           page_bytes, stride_bt, nsplit, page, ld_dbg};
  cudaStream_t s = (cudaStream_t)stream;
#define DISPATCH(R_, BN_)                                                             \
  if (rank == R_ && bn == BN_) return mode == 0 ? launch<R_, BN_, 0>(p, nreq, s) : mode == 2 ? launch<R_, BN_, 2>(p, nreq, s) : mode == 3 ? launch<R_, BN_, 3>(p, nreq, s) : launch<R_, BN_, 1>(p, nreq, s);
#ifdef ONLY768
  if (rank == 768 && bn == 32) return launch<768, 32, 1>(p, nreq, s);
#else
  DISPATCH(768, 32)
  DISPATCH(256, 32)
#endif
#undef DISPATCH
  return -1;
}

extern "C" int mla_int8_smem(int rank, int bn) {
  if (rank == 768 && bn == 32) return Cfg<768, 32>::SMEM;
  if (rank == 256 && bn == 32) return Cfg<256, 32>::SMEM;
  return -1;
}
