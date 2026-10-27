// FAIE2-FN: FAIE2 engine for Qwen3.8-Flash-Next (GGUF arch "qwen4exp"), two RTX 5090s (sm_120), Windows native.
//
// Every weight of shard 1 (47.3 GB, 14 GGUF formats) stays resident in VRAM as its own GGUF blocks, byte-permuted into
// 16-row tiles at load (see "tiled GEMV"), split by layer across the two cards. The 28.8 GB n-gram table (shard 2, read 16 rows per token by layer 1) lives in pinned host
// RAM that GPU0 reads directly over PCIe, so no CPU core touches it. Between the cards the residual stream (40 KB per
// token) goes GPU0 -> pinned RAM -> GPU1 by the copy engines; the CPU only enqueues work.
//
// Math: transformers models/qwen4_exp/modeling_qwen4_exp.py (the official definition), run in fp32 on dequantized
// GGUF values. Dequantization follows ggml-quants.c exactly (tables in fn_tables.cuh, MIT, the ggml authors).
// GGUF storage conventions: RMSNorm weights are stored as 1 + w, ssm_a = -exp(A_log), DeltaNet value heads are tiled
// [v][k] (value head h reads key head h % 16), the DeltaNet gated norm weight is plain.

#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "fn_tables.cuh"

#define FN_API extern "C" __declspec(dllexport)

// ---------------------------------------------------------------------------------------------
// model constants (asserted against the GGUF at load time)

constexpr int D       = 2560;              // hidden
constexpr int NL      = 48;                // layers
constexpr int HC      = 4;                 // hyper-connection streams
constexpr int HCD     = HC * D;            // 10240 residual width
constexpr int HCR     = 320;               // hyper-connection low rank
constexpr int NE      = 512;               // routed experts
constexpr int NEU     = 10;                // experts per token
constexpr int FFE     = 640;               // expert / shared-expert hidden
constexpr int NH      = 24;                // attention q heads
constexpr int NKV     = 2;                 // attention kv heads
constexpr int HD      = 256;               // attention head dim
constexpr int QG      = 2 * HD;            // q_proj rows per head: query then output gate
constexpr int NROT    = 64;                // rotary dims (partial rope)
constexpr int DS      = 128;               // DeltaNet head dim
constexpr int NKH     = 16;                // DeltaNet key heads
constexpr int NVH     = 48;                // DeltaNet value heads
constexpr int DINNER  = NVH * DS;          // 6144
constexpr int CONVCH  = 2 * NKH * DS + DINNER;  // 10240
constexpr int IXH     = 4;                 // QSA indexer query heads
constexpr int IXD     = 128;               // QSA indexer head dim
constexpr int IXR     = 4;                 // tokens per compressed block
constexpr int IXTOP   = 2048 / IXR;        // blocks kept per query (512)
constexpr int VOCAB   = 248320;
constexpr int PLE_LAYER = 1;               // 0-indexed layer with the per-layer n-gram embedding
constexpr int PLE_NH  = 16;                // n-gram heads: 8 bigram + 8 trigram
constexpr int PLE_HD  = 160;               // values per n-gram head
constexpr int PLE_EOS = 248044;
constexpr int PLE_HIST = 9;                // dilated conv state: (kernel 4 - 1) * dilation 3
constexpr int MAXT    = 8;                 // tokens per forward
constexpr float EPS   = 1e-6f;
constexpr float ROPE_BASE = 10000000.0f;
constexpr int ATT_CH  = 128;               // keys per attention tile
constexpr int NSPLIT  = 32;                // attention splits per (token, head); each split walks its share in tiles
constexpr int LRS     = 336;               // row stride of the stacked hyper-connection down + inject output (21 tiles)
constexpr int RSG     = 528;               // row stride of the stacked router + shared-gate output (33 tiles)
constexpr int IXQK    = 640;               // row stride of the stacked indexer q + k output

static bool is_attn_layer(int il) { return (il + 1) % 4 == 0; }

enum : uint32_t {
    T_F32 = 0, T_F16 = 1, T_Q8_0 = 8, T_Q4_K = 12, T_Q5_K = 13, T_Q6_K = 14, T_IQ2_XXS = 16, T_IQ2_XS = 17,
    T_IQ3_XXS = 18, T_IQ4_NL = 20, T_IQ3_S = 21, T_IQ2_S = 22, T_IQ4_XS = 23, T_BF16 = 30, T_Q2_0 = 42,
};

// values per block and bytes per block of each storage type; false for types this engine does not read
static bool type_info(uint32_t t, int & qk, int & bytes) {
    switch (t) {
        case T_F32:     qk = 1;   bytes = 4;   return true;
        case T_F16:     qk = 1;   bytes = 2;   return true;
        case T_BF16:    qk = 1;   bytes = 2;   return true;
        case T_Q8_0:    qk = 32;  bytes = 34;  return true;
        case T_Q4_K:    qk = 256; bytes = 144; return true;
        case T_Q5_K:    qk = 256; bytes = 176; return true;
        case T_Q6_K:    qk = 256; bytes = 210; return true;
        case T_IQ2_XXS: qk = 256; bytes = 66;  return true;
        case T_IQ2_XS:  qk = 256; bytes = 74;  return true;
        case T_IQ2_S:   qk = 256; bytes = 82;  return true;
        case T_IQ3_XXS: qk = 256; bytes = 98;  return true;
        case T_IQ3_S:   qk = 256; bytes = 110; return true;
        case T_IQ4_NL:  qk = 32;  bytes = 18;  return true;
        case T_IQ4_XS:  qk = 256; bytes = 136; return true;
        case T_Q2_0:    qk = 64;  bytes = 18;  return true;
        default: return false;
    }
}

static uint64_t row_bytes(uint32_t t, uint64_t cols) {
    int qk, b;
    if (!type_info(t, qk, b)) return 0;
    return cols / qk * b;
}

// ---------------------------------------------------------------------------------------------
// errors

static std::string g_err;

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    g_err = std::string(#x) + " -> " + cudaGetErrorString(e_); return -1; } } while (0)
#define FAIL(msg) do { g_err = (msg); return -1; } while (0)

// ---------------------------------------------------------------------------------------------
// GGUF reader (memory-mapped)

struct GTensor {
    uint32_t type = 0;
    uint32_t nd = 0;
    uint64_t ne[4] = {1, 1, 1, 1};
    uint64_t offset = 0;
    uint64_t nbytes = 0;
    uint64_t file_offset = 0;
    const uint8_t * host = nullptr;
};

struct Gguf {
    HANDLE hf = INVALID_HANDLE_VALUE;
    HANDLE hm = nullptr;
    const uint8_t * base = nullptr;
    uint64_t size = 0;
    uint64_t alignment = 32;
    std::unordered_map<std::string, GTensor> tensors;
    std::unordered_map<std::string, double> nums;
    std::unordered_map<std::string, std::vector<int64_t>> iarrays;

    ~Gguf() {
        if (base) UnmapViewOfFile(base);
        if (hm) CloseHandle(hm);
        if (hf != INVALID_HANDLE_VALUE) CloseHandle(hf);
    }
};

struct Cursor {
    const uint8_t * p;
    const uint8_t * end;
    bool ok = true;
    template <typename T> T get() {
        T v{};
        if (p + sizeof(T) > end) { ok = false; return v; }
        memcpy(&v, p, sizeof(T));
        p += sizeof(T);
        return v;
    }
    std::string str() {
        uint64_t n = get<uint64_t>();
        if (!ok || p + n > end) { ok = false; return {}; }
        std::string s((const char *) p, n);
        p += n;
        return s;
    }
};

static size_t gguf_scalar_size(uint32_t t) {
    switch (t) {
        case 0: case 1: case 7: return 1;
        case 2: case 3: return 2;
        case 4: case 5: case 6: return 4;
        case 10: case 11: case 12: return 8;
        default: return 0;
    }
}

static double gguf_scalar_value(Cursor & c, uint32_t t) {
    switch (t) {
        case 0: return c.get<uint8_t>();
        case 1: return c.get<int8_t>();
        case 2: return c.get<uint16_t>();
        case 3: return c.get<int16_t>();
        case 4: return c.get<uint32_t>();
        case 5: return c.get<int32_t>();
        case 6: return c.get<float>();
        case 7: return c.get<uint8_t>();
        case 10: return (double) c.get<uint64_t>();
        case 11: return (double) c.get<int64_t>();
        case 12: return c.get<double>();
        default: c.ok = false; return 0;
    }
}

static int gguf_open(Gguf & g, const char * path) {
    g.hf = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (g.hf == INVALID_HANDLE_VALUE) FAIL(std::string("cannot open ") + path);
    LARGE_INTEGER sz;
    GetFileSizeEx(g.hf, &sz);
    g.size = (uint64_t) sz.QuadPart;
    g.hm = CreateFileMappingA(g.hf, nullptr, PAGE_READONLY, 0, 0, nullptr);
    if (!g.hm) FAIL("CreateFileMapping failed");
    g.base = (const uint8_t *) MapViewOfFile(g.hm, FILE_MAP_READ, 0, 0, 0);
    if (!g.base) FAIL("MapViewOfFile failed");

    Cursor c{g.base, g.base + g.size};
    if (c.get<uint32_t>() != 0x46554747u) FAIL("not a GGUF file");
    const uint32_t version = c.get<uint32_t>();
    if (version < 2) FAIL("GGUF version < 2 not supported");
    const uint64_t n_tensors = c.get<uint64_t>();
    const uint64_t n_kv = c.get<uint64_t>();

    for (uint64_t i = 0; i < n_kv && c.ok; ++i) {
        std::string key = c.str();
        uint32_t t = c.get<uint32_t>();
        if (t == 8) {
            c.str();
        } else if (t == 9) {
            uint32_t et = c.get<uint32_t>();
            uint64_t n = c.get<uint64_t>();
            if (et == 8) {
                for (uint64_t j = 0; j < n && c.ok; ++j) c.str();
            } else if (et <= 5 || et == 10 || et == 11) {
                auto & v = g.iarrays[key];
                for (uint64_t j = 0; j < n && c.ok; ++j) {
                    if (et == 10 || et == 11) v.push_back(c.get<int64_t>());  // exact: no trip through double
                    else v.push_back((int64_t) gguf_scalar_value(c, et));
                }
            } else {
                size_t es = gguf_scalar_size(et);
                if (!es || c.p + es * n > c.end) FAIL("bad GGUF array");
                c.p += es * n;
            }
        } else {
            double v = gguf_scalar_value(c, t);
            g.nums[key] = v;
            if (key == "general.alignment") g.alignment = (uint64_t) v;
        }
    }
    std::vector<std::pair<std::string, GTensor>> infos;
    for (uint64_t i = 0; i < n_tensors && c.ok; ++i) {
        std::string name = c.str();
        GTensor t;
        t.nd = c.get<uint32_t>();
        for (uint32_t d = 0; d < t.nd; ++d) t.ne[d] = c.get<uint64_t>();
        t.type = c.get<uint32_t>();
        t.offset = c.get<uint64_t>();
        infos.emplace_back(std::move(name), t);
    }
    if (!c.ok) FAIL("truncated GGUF header");
    uint64_t pos = (uint64_t) (c.p - g.base);
    const uint64_t data_start = (pos + g.alignment - 1) / g.alignment * g.alignment;
    for (auto & [name, t] : infos) {
        t.nbytes = row_bytes(t.type, t.ne[0]) * t.ne[1] * t.ne[2] * t.ne[3];
        t.file_offset = data_start + t.offset;
        t.host = g.base + t.file_offset;
        if (t.file_offset + t.nbytes > g.size) FAIL("tensor " + name + " runs past end of file");
        g.tensors[name] = t;
    }
    return 0;
}

// ---------------------------------------------------------------------------------------------
// device helpers

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}

// result is returned to every thread of the block
template <int NW> __device__ __forceinline__ float block_sum(float v, float * red) {
    v = warp_sum(v);
    const int w = threadIdx.x >> 5, l = threadIdx.x & 31;
    __syncthreads();
    if (l == 0) red[w] = v;
    __syncthreads();
    float r = l < NW ? red[l] : 0.0f;
    return warp_sum(r);
}

template <int NW> __device__ __forceinline__ float block_max(float v, float * red) {
    v = warp_max(v);
    const int w = threadIdx.x >> 5, l = threadIdx.x & 31;
    __syncthreads();
    if (l == 0) red[w] = v;
    __syncthreads();
    float r = l < NW ? red[l] : -INFINITY;
    return warp_max(r);
}

__device__ __forceinline__ float silu(float x) { return x / (1.0f + expf(-x)); }

// Sums of products written out as the FMAs nvcc formed for them in the first kernels (read off their SASS): a fused
// kernel that reorganizes the code must not let the compiler contract them differently, or its bits change.
// a0 b0 + a1 b1 + a2 b2 + a3 b3: the second product rounded, then the first, third and fourth fused in
__device__ __forceinline__ float dot4c(float a0, float b0, float a1, float b1, float a2, float b2, float a3, float b3) {
    return __fmaf_rn(a3, b3, __fmaf_rn(a2, b2, __fmaf_rn(a0, b0, __fmul_rn(a1, b1))));
}
__device__ __forceinline__ float sigm(float x) { return 1.0f / (1.0f + expf(-x)); }

// Programmatic dependent launch. Every forward kernel is launched with programmatic stream serialization: it may start
// while the previous kernel finishes (its blocks are already resident when that one ends), GEMVs stage their first
// weight units (weights never change), then pdl_wait() for the previous kernel to complete before touching
// activations. Every kernel calls pdl_trigger() only after its wait (for the kernels a GEMV follows, once their own
// loads have arrived, else they queue behind the GEMV's weight prefetch). So a kernel that starts early runs beside
// at most its direct predecessor, and every kernel two or more back has completed. The one exception is a GEMV
// launched early (see gemv()): it reads only data complete before its predecessor started, triggers at once and
// waits before it exits, so its completion still implies its predecessor's.
__device__ __forceinline__ void pdl_wait() { asm volatile("griddepcontrol.wait;\n" ::: "memory"); }
__device__ __forceinline__ void pdl_trigger() { asm volatile("griddepcontrol.launch_dependents;\n" ::: "memory"); }
// nvcc treats loads through const __restrict__ pointers (and __ldg) as reads of memory that never changes and may move
// them above the wait's memory clobber (it did: k_tmoe_gu read sel before its wait). A pointer to data an earlier
// kernel of this forward writes goes through after_wait() once the wait has returned: its value then comes from an asm
// statement ordered after the wait, so no load through it can issue earlier. flashnext/pdl_audit_fn.py lists the
// loads each kernel issues before its wait.
template <typename P> __device__ __forceinline__ P after_wait(P p) {
    asm volatile("" : "+l"(p));
    return p;
}

// Every GGUF block of these formats is an even number of bytes and rows start at even offsets (device rows, tiled
// units, the n-gram table), so 16-bit loads are always aligned.
// Every GGUF block of these formats is an even number of bytes and rows start at even offsets (device rows, tiled
// units, the n-gram table), so 16-bit loads are always aligned.
__device__ __forceinline__ uint32_t ld16(const uint8_t * p) { return *(const uint16_t *) p; }
__device__ __forceinline__ uint32_t ld32(const uint8_t * p) { return ld16(p) | (ld16(p + 2) << 16); }
// 4-byte aligned word (Q4_K/Q5_K blocks are 16-byte multiples, IQ4_XS 8-byte, and their quant bytes sit at aligned
// offsets inside the block)
__device__ __forceinline__ uint32_t ld32a(const uint8_t * p) { return *(const uint32_t *) p; }
__device__ __forceinline__ float ldh(const uint8_t * p) { return __half2float(__ushort_as_half((unsigned short) ld16(p))); }
__device__ __forceinline__ float bf16f(uint32_t u) { return __uint_as_float(u << 16); }

// ---------------------------------------------------------------------------------------------
// dequantization (ggml-quants.c arithmetic)
//
// dqv<T>(row, g, f) decodes the g-th group of 32 values of a row and hands them to f(j, v[j], v[j+1], v[j+2], v[j+3])
// in eight runs of four (j a multiple of 4, in an order of the format's choosing), so a GEMV consumes values as they
// appear instead of holding 32; dq32<T>(row, g, v) is the same decode written to an array.
//
// Small integers become floats without I2F (a quarter-rate conversion): with the integer u (< 2^23) in the low
// mantissa bits of 2^23, the float is 2^23 + u exactly, and subtracting 2^23 (+ the format's offset) is exact.

__device__ __forceinline__ float u2f(uint32_t u) { return __int_as_float(0x4B000000u | u) - 8388608.0f; }
// 0x4B000000 from constant memory: as a literal the compiler makes it PRMT's immediate and moves every byte selector
// into a register first; from the constant bank the selector is the immediate
__constant__ uint32_t c_mag = 0x4B000000u;
// bits 0x4B000000 | byte k of w
__device__ __forceinline__ uint32_t byte_m(uint32_t w, int k) { return __byte_perm(w, c_mag, 0x7440 + k); }
// byte k of w as float, minus `off` (off = 0, or 128 for a biased int8)
__device__ __forceinline__ float byte_f(uint32_t w, int k, float off) { return __int_as_float(byte_m(w, k)) - (8388608.0f + off); }
// ksigns_iq2xs[i]: the 7 sign bits plus their parity in bit 7 (checked equal to the table for all 128 entries)
__device__ __forceinline__ uint32_t ksign(uint32_t i) { return i | ((__popc(i) & 1u) << 7); }

// The IQ codebooks (1-8 KB) as 32-bit words. GEMV kernels copy them to shared memory: scattered lookups into a small
// table served from shared memory cost a few bank conflicts, from L1 about one cycle per cache line touched.
template <uint32_t TY> struct GRID { static constexpr int words = 0; };
template <> struct GRID<T_IQ2_XXS> { static constexpr int words = 512;  __device__ static const uint32_t * tab() { return (const uint32_t *) iq2xxs_grid; } };
template <> struct GRID<T_IQ2_XS>  { static constexpr int words = 1024; __device__ static const uint32_t * tab() { return (const uint32_t *) iq2xs_grid; } };
template <> struct GRID<T_IQ2_S>   { static constexpr int words = 2048; __device__ static const uint32_t * tab() { return (const uint32_t *) iq2s_grid; } };
template <> struct GRID<T_IQ3_XXS> { static constexpr int words = 256;  __device__ static const uint32_t * tab() { return iq3xxs_grid; } };
template <> struct GRID<T_IQ3_S>   { static constexpr int words = 512;  __device__ static const uint32_t * tab() { return iq3s_grid; } };
static int grid_words(uint32_t t) {
    switch (t) {
        case T_IQ2_XXS: return 512;
        case T_IQ2_XS:  return 1024;
        case T_IQ2_S:   return 2048;
        case T_IQ3_XXS: return 256;
        case T_IQ3_S:   return 512;
        default: return 0;
    }
}
// copy TY's codebook (if any) into shared memory; returns the table the decoder should use
template <uint32_t TY> __device__ __forceinline__ const uint32_t * stage_grid(uint32_t * smem_tab) {
    if constexpr (GRID<TY>::words > 0) {
        for (int i = threadIdx.x; i < GRID<TY>::words; i += blockDim.x) smem_tab[i] = GRID<TY>::tab()[i];
        __syncthreads();
        return smem_tab;
    } else {
        return nullptr;
    }
}

// d * q for the integer q held as the float 2^23 + q (bits 0x4B000000 | q), in one FMA: d * (2^23 + q) + dc with
// dc = -d * 2^23 (exact: a power-of-two scaling) is d * q rounded once, the same bits as d * (float) q
__device__ __forceinline__ float mulq(float d, float dc, uint32_t bits) { return __fmaf_rn(d, __int_as_float(bits), dc); }
__device__ __forceinline__ float bias23(float d) { return -d * 8388608.0f; }

// v * (sign bit j of `signs` ? -1 : 1), exactly: a sign-bit flip
__device__ __forceinline__ float flip(float v, uint32_t signs, int j) {
    return __uint_as_float(__float_as_uint(v) ^ ((signs << (31 - j)) & 0x80000000u));
}

// kvalues_iq4nl + 128 as bytes, 4 per word (entries 0-3, 4-7, 8-11, 12-15)
constexpr uint32_t KV4_0 = 0x3F2D1801u, KV4_1 = 0x766A5D4Fu, KV4_2 = 0xA6998D81u, KV4_3 = 0xF1D9C5B5u;
// the 8 nibbles of w (byte b: low nibble = value b, high nibble = value b + 16 of its group of 32) looked up in
// kvalues_iq4nl + 128: lo = bytes of values 0..3, hi = values 16..19 (relative to w's first byte)
__device__ __forceinline__ void iq4_lookup(uint32_t w, uint32_t & lo, uint32_t & hi) {
    const uint32_t idx = w & 0x77777777u, sel = 0x32103210u | ((w & 0x88888888u) >> 1);
    uint32_t t[2];
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const uint32_t a = __byte_perm(KV4_0, KV4_1, idx >> (16 * i)), b = __byte_perm(KV4_2, KV4_3, idx >> (16 * i));
        t[i] = __byte_perm(a, b, sel >> (16 * i));  // nibble order: value 0, 16, 1, 17 | 2, 18, 3, 19
    }
    lo = __byte_perm(t[0], t[1], 0x6420);
    hi = __byte_perm(t[0], t[1], 0x7531);
}

__device__ __forceinline__ void scale_min_k4(int j, const uint8_t * q, float & d, float & m) {
    if (j < 4) {
        d = (float) (q[j] & 63);
        m = (float) (q[j + 4] & 63);
    } else {
        d = (float) ((q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4));
        m = (float) ((q[j + 4] >> 4) | ((q[j] >> 6) << 4));
    }
}

template <uint32_t TY> struct DQ;

template <> struct DQ<T_BF16> {
    template <typename F>
    __device__ __forceinline__ static void run(const uint8_t * __restrict__ row, int g, F && f, const uint32_t * = nullptr) {
        const uint8_t * p = row + g * 64;
#pragma unroll
        for (int j = 0; j < 32; j += 4)
            f(j, bf16f(ld16(p + 2 * j)), bf16f(ld16(p + 2 * j + 2)), bf16f(ld16(p + 2 * j + 4)), bf16f(ld16(p + 2 * j + 6)));
    }
};

template <> struct DQ<T_Q8_0> {
    template <typename F>
    __device__ __forceinline__ static void run(const uint8_t * __restrict__ row, int g, F && f, const uint32_t * = nullptr) {
        const uint8_t * b = row + g * 34;
        const float d = ldh(b);
#pragma unroll
        for (int j = 0; j < 32; j += 4) {
            const uint32_t w0 = ld16(b + 2 + j) ^ 0x8080u, w1 = ld16(b + 4 + j) ^ 0x8080u;
            f(j, byte_f(w0, 0, 128.0f) * d, byte_f(w0, 1, 128.0f) * d, byte_f(w1, 0, 128.0f) * d, byte_f(w1, 1, 128.0f) * d);
        }
    }
};

template <> struct DQ<T_IQ4_NL> {
    template <typename F>
    __device__ __forceinline__ static void run(const uint8_t * __restrict__ row, int g, F && f, const uint32_t * = nullptr) {
        const uint8_t * b = row + g * 18;
        const float d = ldh(b);
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            uint32_t lo, hi;
            iq4_lookup(ld32(b + 2 + 4 * k), lo, hi);
            f(4 * k, d * byte_f(lo, 0, 128.0f), d * byte_f(lo, 1, 128.0f), d * byte_f(lo, 2, 128.0f), d * byte_f(lo, 3, 128.0f));
            f(16 + 4 * k, d * byte_f(hi, 0, 128.0f), d * byte_f(hi, 1, 128.0f), d * byte_f(hi, 2, 128.0f),
              d * byte_f(hi, 3, 128.0f));
        }
    }
};

template <> struct DQ<T_Q2_0> {
    template <typename F>
    __device__ __forceinline__ static void run(const uint8_t * __restrict__ row, int g, F && f, const uint32_t * = nullptr) {
        const uint8_t * b = row + (g >> 1) * 18;
        const float d = ldh(b);
        const uint8_t * qs = b + 2 + (g & 1) * 8;
        auto q = [&](uint32_t w, int j) { return (__int_as_float(0x4B000000u | ((w >> (2 * j)) & 3u)) - 8388609.0f) * d; };
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            const uint32_t w = ld16(qs + 2 * k);  // 8 values of 2 bits
            f(8 * k, q(w, 0), q(w, 1), q(w, 2), q(w, 3));
            f(8 * k + 4, q(w, 4), q(w, 5), q(w, 6), q(w, 7));
        }
    }
};

// 8 grid bytes (two words) times dl with signs: runs j0 and j0 + 4
template <typename F>
__device__ __forceinline__ void grid8(F && f, int j0, uint32_t g0, uint32_t g1, float dl, uint32_t signs) {
    const float dc = bias23(dl);
    auto q = [&](uint32_t g, int k, int j) { return flip(mulq(dl, dc, byte_m(g, k)), signs, j); };
    f(j0, q(g0, 0, 0), q(g0, 1, 1), q(g0, 2, 2), q(g0, 3, 3));
    f(j0 + 4, q(g1, 0, 4), q(g1, 1, 5), q(g1, 2, 6), q(g1, 3, 7));
}

template <> struct DQ<T_IQ2_XXS> {
    template <typename F>
    __device__ __forceinline__ static void run(const uint8_t * __restrict__ row, int g, F && f, const uint32_t * __restrict__ tab) {
        const uint8_t * b = row + (g >> 3) * 66;
        const int ib = g & 7;
        const float d = ldh(b);
        const uint32_t a0 = ld32(b + 2 + 8 * ib), a1 = ld32(b + 2 + 8 * ib + 4);
        const float db = d * (0.5f + u2f(a1 >> 28)) * 0.25f;
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const uint2 grid = *(const uint2 *) &tab[2 * ((a0 >> (8 * l)) & 255)];
            grid8(f, 8 * l, grid.x, grid.y, db, ksign((a1 >> (7 * l)) & 127));
        }
    }
};

template <> struct DQ<T_IQ2_XS> {
    template <typename F>
    __device__ __forceinline__ static void run(const uint8_t * __restrict__ row, int g, F && f, const uint32_t * __restrict__ tab) {
        const uint8_t * b = row + (g >> 3) * 74;
        const int ib = g & 7;
        const float d = ldh(b);
        const uint32_t sc = b[2 + 64 + ib];
        const float db0 = d * (0.5f + u2f(sc & 0xF)) * 0.25f;
        const float db1 = d * (0.5f + u2f(sc >> 4)) * 0.25f;
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const uint32_t q = ld16(b + 2 + 2 * (4 * ib + l));
            const uint2 grid = *(const uint2 *) &tab[2 * (q & 511)];
            grid8(f, 8 * l, grid.x, grid.y, l < 2 ? db0 : db1, ksign(q >> 9));
        }
    }
};

template <> struct DQ<T_IQ2_S> {
    template <typename F>
    __device__ __forceinline__ static void run(const uint8_t * __restrict__ row, int g, F && f, const uint32_t * __restrict__ tab) {
        const uint8_t * b = row + (g >> 3) * 82;
        const int ib = g & 7;
        const float d = ldh(b);
        const uint32_t qs = ld32(b + 2 + 4 * ib), signs = ld32(b + 2 + 32 + 4 * ib);
        const uint32_t qh = b[2 + 64 + ib];
        const uint32_t sc = b[2 + 64 + 8 + ib];
        const float db0 = d * (0.5f + u2f(sc & 0xF)) * 0.25f;
        const float db1 = d * (0.5f + u2f(sc >> 4)) * 0.25f;
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const uint2 grid = *(const uint2 *) &tab[2 * (((qs >> (8 * l)) & 255) | ((qh << (8 - 2 * l)) & 0x300))];
            grid8(f, 8 * l, grid.x, grid.y, l < 2 ? db0 : db1, signs >> (8 * l));
        }
    }
};

template <> struct DQ<T_IQ3_XXS> {
    template <typename F>
    __device__ __forceinline__ static void run(const uint8_t * __restrict__ row, int g, F && f, const uint32_t * __restrict__ tab) {
        const uint8_t * b = row + (g >> 3) * 98;
        const int ib = g & 7;
        const float d = ldh(b);
        const uint32_t q0 = ld32(b + 2 + 8 * ib), q1 = ld32(b + 2 + 8 * ib + 4);
        const uint32_t aux = ld32(b + 2 + 64 + 4 * ib);
        const float db = d * (0.5f + u2f(aux >> 28)) * 0.5f;
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const uint32_t qq = l < 2 ? q0 : q1, s = 16 * (l & 1);
            grid8(f, 8 * l, tab[(qq >> s) & 255], tab[(qq >> (s + 8)) & 255], db, ksign((aux >> (7 * l)) & 127));
        }
    }
};

template <> struct DQ<T_IQ3_S> {
    template <typename F>
    __device__ __forceinline__ static void run(const uint8_t * __restrict__ row, int g, F && f, const uint32_t * __restrict__ tab) {
        const uint8_t * b = row + (g >> 3) * 110;
        const int ib = g & 7;
        const float d = ldh(b);
        const uint32_t q0 = ld32(b + 2 + 8 * ib), q1 = ld32(b + 2 + 8 * ib + 4);
        const uint32_t qh = b[2 + 64 + ib];
        const uint32_t signs = ld32(b + 2 + 64 + 8 + 4 * ib);
        const uint32_t scb = b[2 + 64 + 8 + 32 + (ib >> 1)];
        const float db = d * u2f(1 + 2 * ((scb >> (4 * (ib & 1))) & 0xF));
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const uint32_t qq = l < 2 ? q0 : q1, s = 16 * (l & 1);
            grid8(f, 8 * l, tab[((qq >> s) & 255) | ((qh << (8 - 2 * l)) & 256)],
                  tab[((qq >> (s + 8)) & 255) | ((qh << (7 - 2 * l)) & 256)], db, signs >> (8 * l));
        }
    }
};

template <> struct DQ<T_IQ4_XS> {
    template <typename F>
    __device__ __forceinline__ static void run(const uint8_t * __restrict__ row, int g, F && f, const uint32_t * = nullptr) {
        const uint8_t * b = row + (g >> 3) * 136;
        const int ib = g & 7;
        const float d = ldh(b);
        const uint32_t sh = ld16(b + 2);
        const int ls = ((b[4 + (ib >> 1)] >> (4 * (ib & 1))) & 0xF) | (((sh >> (2 * ib)) & 3) << 4);
        const float dl = d * (u2f(ls) - 32.0f);
        const uint8_t * qs = b + 8 + 16 * ib;
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            uint32_t lo, hi;
            iq4_lookup(ld32a(qs + 4 * k), lo, hi);
            f(4 * k, dl * byte_f(lo, 0, 128.0f), dl * byte_f(lo, 1, 128.0f), dl * byte_f(lo, 2, 128.0f),
              dl * byte_f(lo, 3, 128.0f));
            f(16 + 4 * k, dl * byte_f(hi, 0, 128.0f), dl * byte_f(hi, 1, 128.0f), dl * byte_f(hi, 2, 128.0f),
              dl * byte_f(hi, 3, 128.0f));
        }
    }
};

template <> struct DQ<T_Q4_K> {
    template <typename F>
    __device__ __forceinline__ static void run(const uint8_t * __restrict__ row, int g, F && f, const uint32_t * = nullptr) {
        const uint8_t * b = row + (g >> 3) * 144;
        const int ib = g & 7;
        const float d = ldh(b), dmin = ldh(b + 2);
        float sc, m;
        scale_min_k4(ib, b + 4, sc, m);
        const float d1 = d * sc, m1 = dmin * m, d1c = bias23(d1);
        const uint8_t * q = b + 16 + 32 * (ib >> 1);
        const int sh = 4 * (ib & 1);
        auto v = [&](uint32_t n) { return __fsub_rn(mulq(d1, d1c, 0x4B000000u | (n & 0xF)), m1); };
#pragma unroll
        for (int j = 0; j < 32; j += 4) {
            const uint32_t w = ld32a(q + j) >> sh;
            f(j, v(w), v(w >> 8), v(w >> 16), v(w >> 24));
        }
    }
};

template <> struct DQ<T_Q5_K> {
    template <typename F>
    __device__ __forceinline__ static void run(const uint8_t * __restrict__ row, int g, F && f, const uint32_t * = nullptr) {
        const uint8_t * b = row + (g >> 3) * 176;
        const int ib = g & 7;
        const float d = ldh(b), dmin = ldh(b + 2);
        float sc, m;
        scale_min_k4(ib, b + 4, sc, m);
        const float d1 = d * sc, m1 = dmin * m, d1c = bias23(d1);
        const uint8_t * qh = b + 16;
        const uint8_t * ql = b + 48 + 32 * (ib >> 1);
        const int sh = 4 * (ib & 1);
        auto v = [&](uint32_t w, uint32_t h) { return __fsub_rn(mulq(d1, d1c, 0x4B000000u | (w & 0xF) | ((h & 1) << 4)), m1); };
#pragma unroll
        for (int j = 0; j < 32; j += 4) {
            const uint32_t w = ld32a(ql + j) >> sh, h = ld32a(qh + j) >> ib;
            f(j, v(w, h), v(w >> 8, h >> 8), v(w >> 16, h >> 16), v(w >> 24, h >> 24));
        }
    }
};

template <> struct DQ<T_Q6_K> {
    template <typename F>
    __device__ __forceinline__ static void run(const uint8_t * __restrict__ row, int g, F && f, const uint32_t * = nullptr) {
        const uint8_t * b = row + (g >> 3) * 210;
        const int ib = g & 7, n = ib >> 2, qu = ib & 3;
        const uint8_t * ql = b + 64 * n + ((qu & 1) ? 32 : 0);
        const uint8_t * qh = b + 128 + 32 * n;
        const int8_t * sc = (const int8_t *) (b + 192 + 8 * n);
        const float d = ldh(b + 208);
        const int lsh = (qu & 2) ? 4 : 0, hsh = 2 * qu;
        // ds = d * sc is exact (11 + 7 significant bits), so ds * 3 * 2^22 is too, and the integer q - 32 rides on
        // 1.5 * 2^23: bits 0x4B3FFFE0 + q are the float 1.5 * 2^23 + (q - 32)
        const float ds0 = d * (float) sc[2 * qu], ds1 = d * (float) sc[1 + 2 * qu];
        const float dc0 = -ds0 * 12582912.0f, dc1 = -ds1 * 12582912.0f;
        auto v = [&](float ds, float dc, uint32_t w, uint32_t h) {
            return __fmaf_rn(ds, __int_as_float(0x4B3FFFE0u + ((w & 0xF) | ((h & 3) << 4))), dc);
        };
#pragma unroll
        for (int j = 0; j < 32; j += 4) {
            const uint32_t w0 = ld16(ql + j) >> lsh, w1 = ld16(ql + j + 2) >> lsh;
            const uint32_t h0 = ld16(qh + j) >> hsh, h1 = ld16(qh + j + 2) >> hsh;
            const float ds = j < 16 ? ds0 : ds1, dc = j < 16 ? dc0 : dc1;
            f(j, v(ds, dc, w0, h0), v(ds, dc, w0 >> 8, h0 >> 8), v(ds, dc, w1, h1), v(ds, dc, w1 >> 8, h1 >> 8));
        }
    }
};

// the global codebook of a grid format (nullptr for the others)
template <uint32_t TY> __device__ __forceinline__ const uint32_t * gtab() {
    if constexpr (GRID<TY>::words > 0) return GRID<TY>::tab();
    else return nullptr;
}

template <uint32_t TY> __device__ __forceinline__ void dq32(const uint8_t * __restrict__ row, int g, float * v) {
    DQ<TY>::run(row, g, [&](int j, float a, float b, float c, float d) {
        v[j] = a;
        v[j + 1] = b;
        v[j + 2] = c;
        v[j + 3] = d;
    }, gtab<TY>());
}

// ---------------------------------------------------------------------------------------------
// tiled GEMV
//
// At load every weight is repacked into units of 16 rows x one chunk of CV values (256 when the row length allows,
// else 128/64/32; BF16 always 64): unit (tile T, chunk c) starts at byte (T * nch + c) * 16 * cb and holds the 16
// rows' cb bytes back to back (cb = bytes of CV values). A pure permutation of the GGUF bytes (rows padded to a
// multiple of 16 with zeros), so dq32 decodes a row of a unit exactly like the same span of the original row.
// Expert tensors are stacks of 640- or 2560-row matrices, so a tile never straddles two experts.
//
// A warp owns a contiguous range of a tile's chunks (split si of S = wpt x ks). It stages each unit in shared memory
// with cp.async (double buffered), then lane = (row lane & 15, half lane >> 4 of the unit's 32-value groups) decodes
// with dq32 and reads x with __ldg float4. The halves meet by shfl_xor 16, the wpt warps of a tile in shared memory
// (warp order), the ks blocks of a tile through a partial buffer summed in split order. Every sum has a fixed order
// that depends on the weight's shape only, never on the number of tokens, so a token's result is bit-identical in
// any batch. The dense kernels are persistent: as many blocks as fit the GPU, each walking groups of tb tiles with
// the next unit's copy in flight across tile boundaries (no partial last wave).

template <uint32_t TY> struct QT;
template <> struct QT<T_BF16>    { static constexpr int qk = 1,   bytes = 2;   };
template <> struct QT<T_Q8_0>    { static constexpr int qk = 32,  bytes = 34;  };
template <> struct QT<T_IQ4_NL>  { static constexpr int qk = 32,  bytes = 18;  };
template <> struct QT<T_Q2_0>    { static constexpr int qk = 64,  bytes = 18;  };
template <> struct QT<T_Q4_K>    { static constexpr int qk = 256, bytes = 144; };
template <> struct QT<T_Q5_K>    { static constexpr int qk = 256, bytes = 176; };
template <> struct QT<T_Q6_K>    { static constexpr int qk = 256, bytes = 210; };
template <> struct QT<T_IQ2_XXS> { static constexpr int qk = 256, bytes = 66;  };
template <> struct QT<T_IQ2_XS>  { static constexpr int qk = 256, bytes = 74;  };
template <> struct QT<T_IQ2_S>   { static constexpr int qk = 256, bytes = 82;  };
template <> struct QT<T_IQ3_XXS> { static constexpr int qk = 256, bytes = 98;  };
template <> struct QT<T_IQ3_S>   { static constexpr int qk = 256, bytes = 110; };
template <> struct QT<T_IQ4_XS>  { static constexpr int qk = 256, bytes = 136; };

// bytes of one row of a unit
template <uint32_t TY, int CV> __host__ __device__ constexpr int qcb() { return CV / QT<TY>::qk * QT<TY>::bytes; }

__device__ __forceinline__ void cp16(void * smem, const void * gmem) {
    const unsigned s = (unsigned) __cvta_generic_to_shared(smem);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(s), "l"(gmem));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N> __device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }

// FMA chains of one token over a run of four values (chain k takes values j + k), written out with __fmaf_rn so every
// kernel instantiation (1-token or 8-token, dense or expert) computes the same roundings for the same token
__device__ __forceinline__ void fma4(float (&c)[4], const float * __restrict__ xp, float v0, float v1, float v2, float v3) {
    const float4 q = __ldg((const float4 *) xp);
    c[0] = __fmaf_rn(v0, q.x, c[0]);
    c[1] = __fmaf_rn(v1, q.y, c[1]);
    c[2] = __fmaf_rn(v2, q.z, c[2]);
    c[3] = __fmaf_rn(v3, q.w, c[3]);
}
__device__ __forceinline__ float chains(const float (&c)[4]) { return __fadd_rn(__fadd_rn(c[0], c[1]), __fadd_rn(c[2], c[3])); }

// one run of four values against tokens t < nt (x of token t at xp + t * xs): every x load is issued before the FMAs
// (predicated, not branched, so they overlap); chains of tokens >= nt are left for the caller to ignore
template <int NT>
__device__ __forceinline__ void fma4n(float (&c)[NT][4], const float * __restrict__ xp, int xs, int nt, float v0, float v1,
                                      float v2, float v3) {
    float4 q[NT];
#pragma unroll
    for (int t = 0; t < NT; ++t)  // the 1-token kernel loads unconditionally (a forward has at least one token)
        q[t] = NT == 1 || t < nt ? __ldg((const float4 *) (xp + (size_t) t * xs)) : make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
    for (int t = 0; t < NT; ++t) {
        c[t][0] = __fmaf_rn(v0, q[t].x, c[t][0]);
        c[t][1] = __fmaf_rn(v1, q[t].y, c[t][1]);
        c[t][2] = __fmaf_rn(v2, q[t].z, c[t][2]);
        c[t][3] = __fmaf_rn(v3, q[t].w, c[t][3]);
    }
}

// One warp: a[m][t] += rows (lane & 15) of NM tiled matrices . x[t], over n consecutive chunks. src[m] = the first unit
// of matrix m; x = x[0] at the first chunk (token stride xs). buf = nbuf stages x NM units of shared memory.
template <uint32_t TY, int CV, int NM, int NA>
__device__ __forceinline__ void tile_dot(const uint8_t * const * src, int n, uint8_t * buf, int nbuf,
                                         const float * __restrict__ x, int xs, int nt, float (&a)[NM][NA], int lane,
                                         const uint32_t * __restrict__ tab) {
    constexpr int CB = qcb<TY, CV>(), UB = 16 * CB, GPC = CV / 32, GPH = GPC > 1 ? GPC / 2 : 1;
    auto stage = [&](int j, int sb) {
#pragma unroll
        for (int m = 0; m < NM; ++m) {
            uint8_t * d = buf + (sb * NM + m) * UB;
            const uint8_t * s = src[m] + (size_t) j * UB;
            for (int p = lane; p < CB; p += 32) cp16(d + 16 * p, s + 16 * p);
        }
        cp_commit();
    };
    if (n > 0) stage(0, 0);
    // k_tmoe_down reads sel before this wait (k_route is three kernels back, and the two between trigger only after
    // k_route completed); k_tmoe_gu waits before it reads sel
    pdl_wait();
    x = after_wait(x);
    pdl_trigger();
    for (int j = 0; j < n; ++j) {
        const int cur = nbuf > 1 ? (j & 1) : 0;
        if (nbuf > 1 && j + 1 < n) {
            stage(j + 1, cur ^ 1);  // that buffer was last read in iteration j - 1, closed by its __syncwarp
            cp_wait<1>();
        } else {
            cp_wait<0>();
        }
        __syncwarp();
#pragma unroll 1
        for (int q = 0; q < GPH; ++q) {
            const int g = (lane >> 4) * GPH + q;
            if (g >= GPC) break;
            const float * xg = x + j * CV + g * 32;
#pragma unroll
            for (int m = 0; m < NM; ++m) {
                float c[NA][4] = {};
                DQ<TY>::run(buf + (cur * NM + m) * UB + (lane & 15) * CB, g, [&](int jj, float v0, float v1, float v2, float v3) {
                    fma4n<NA>(c, xg + jj, xs, nt, v0, v1, v2, v3);
                }, tab);
#pragma unroll
                for (int t = 0; t < NA; ++t) a[m][t] = __fadd_rn(a[m][t], chains(c[t]));
            }
        }
        __syncwarp();
        if (nbuf == 1 && j + 1 < n) stage(j + 1, 0);
    }
}

// Finish a tile: halves by shfl_xor 16, the tile's wpt warps through red[warp][16][NT] in warp order (one
// __syncthreads; callers alternate two red buffers between tiles), then row results to y (ks == 1) or to
// part[split][t][ntiles * 16].
template <int NT>
__device__ __forceinline__ void tile_reduce(float (&a)[NT], float * red, int warp, int lane, int tl, int wk, int wpt) {
#pragma unroll
    for (int t = 0; t < NT; ++t) a[t] += __shfl_xor_sync(0xffffffffu, a[t], 16);
    if (wpt > 1) {
        if (lane < 16) {
#pragma unroll
            for (int t = 0; t < NT; ++t) red[(warp * 16 + lane) * NT + t] = a[t];
        }
        __syncthreads();
        if (wk == 0 && lane < 16) {
#pragma unroll
            for (int t = 0; t < NT; ++t) {
                float s = red[(tl * wpt * 16 + lane) * NT + t];
                for (int k = 1; k < wpt; ++k) s += red[((tl * wpt + k) * 16 + lane) * NT + t];
                a[t] = s;
            }
        }
    }
}

template <int NT>
__device__ __forceinline__ void tile_store(float (&a)[NT], float * red, int warp, int lane, int tl, int wk, int wpt,
                                           int tile, int ntiles, int rows, int nt, int ks, float * __restrict__ y, int ys,
                                           int acc, float * __restrict__ part) {
    tile_reduce<NT>(a, red, warp, lane, tl, wk, wpt);
    if (wk != 0 || lane >= 16 || tile >= ntiles) return;
    const int r = tile * 16 + lane;
#pragma unroll
    for (int t = 0; t < NT; ++t) {
        if (t >= nt) break;
        if (ks == 1) {
            if (r < rows) {
                float * o = y + (size_t) t * ys + r;
                *o = acc ? *o + a[t] : a[t];
            }
        } else {
            part[((size_t) blockIdx.y * MAXT + t) * ntiles * 16 + r] = a[t];
        }
    }
}

// Finish a tile of the hyper-connection up weight, stored with rows permuted (hc_perm_row) so tile T holds u[s][c] for
// the 4 streams s of columns c = 4T..4T+3 (row 4 * (c % 4) + s of the tile), and mix instead of storing u, with the
// arithmetic of the mix kernel this replaces: x[t][c] = mean_s sigmoid(u[t][s][c]) * hn[t][s][c], and, from the warp
// holding tile 0 (zinj non-null), inj[t][s] = 2 sigmoid(zinj[t][s] / HC).   ks == 1
// hv[t] = hn[t][s][c] of the lane's row (s = lane % 4, c = 4 * tile + (lane / 4) % 4), loaded when the tile starts.
template <int NT>
__device__ __forceinline__ void mix_prefetch(float (&hv)[NT], const float * __restrict__ hn, int tile, int ntiles, int lane,
                                             int nt) {
    if (tile >= ntiles) return;
    const float * p = hn + (lane & 3) * D + tile * 4 + ((lane >> 2) & 3);
#pragma unroll
    for (int t = 0; t < NT; ++t)
        if (NT == 1 || t < nt) hv[t] = p[(size_t) t * HCD];
}
template <int NT>
__device__ __forceinline__ void tile_mix(float (&a)[NT], const float (&hv)[NT], float * red, int warp, int lane, int tl,
                                         int wk, int wpt, int tile, int ntiles, int nt, const float * __restrict__ zinj,
                                         float * __restrict__ xo, float * __restrict__ injo) {
    tile_reduce<NT>(a, red, warp, lane, tl, wk, wpt);
    if (wk != 0 || tile >= ntiles) return;  // the whole warp stays for the shuffles; lanes < 16 hold the rows
    const int c = tile * 4 + ((lane >> 2) & 3);
#pragma unroll
    for (int t = 0; t < NT; ++t) {
        if (t >= nt) break;
        const float sg = sigm(a[t]);
        float s = 0.0f;
#pragma unroll
        for (int k = 0; k < HC; ++k)
            s += __shfl_sync(0xffffffffu, sg, (lane & ~3) + k) * __shfl_sync(0xffffffffu, hv[t], (lane & ~3) + k);
        if (lane < 16 && (lane & 3) == 0) xo[(size_t) t * D + c] = s / HC;
        if (zinj && tile == 0 && lane < HC) injo[t * HC + lane] = 2.0f * sigm(zinj[t * HC + lane] / HC);
    }
}

// y[t][r] (+)= W[r] . x[t] for t < nt <= NT.  grid (persistent, ks), tb * wpt warps: block x walks tile groups
// blockIdx.x + i * gridDim.x of tb tiles; warp (tl, wk) takes tile group * tb + tl, chunk split blockIdx.y * wpt + wk.
// NT = 1 is the decode instantiation (fewer registers), NT = MAXT serves 2..8 tokens; both run the same per-token ops.
template <uint32_t TY, int CV, int NT>
__global__ void __launch_bounds__(256, 2) k_tgemv(const uint8_t * __restrict__ W, int ntiles, int nch, int rows, int wpt,
                                                  int ks, int nbuf, const float * __restrict__ x, int xs,
                                                  float * __restrict__ y, int ys, int nt, int acc, float * __restrict__ part,
                                                  int early) {
    constexpr int CB = qcb<TY, CV>(), UB = 16 * CB, GPC = CV / 32, GPH = GPC > 1 ? GPC / 2 : 1;
    extern __shared__ __align__(16) uint8_t sm[];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, nw = blockDim.x >> 5, tb = nw / wpt;
    const int tl = warp / wpt, wk = warp % wpt;
    const int S = wpt * ks, si = blockIdx.y * wpt + wk;
    const int c0 = si * nch / S, n = (si + 1) * nch / S - c0;
    const int ngroups = (ntiles + tb - 1) / tb;
    uint8_t * buf = sm + (size_t) warp * nbuf * UB;
    float * red = (float *) (sm + (size_t) nw * nbuf * UB);
    const uint32_t * tab = stage_grid<TY>((uint32_t *) (red + nw * 2 * 16 * MAXT));
    // steps (group gi, chunk j < n); a step stages unit (gi * tb + tl, c0 + j), nothing past the last tile
    auto stage = [&](int gi, int j, int sb) {
        const int tile = gi * tb + tl;
        if (tile < ntiles) {
            const uint8_t * src = W + ((size_t) tile * nch + c0 + j) * UB;
            for (int p = lane; p < CB; p += 32) cp16(buf + sb * UB + 16 * p, src + 16 * p);
        }
        cp_commit();
    };
    int gi = blockIdx.x, sidx = 0;
    if (gi < ngroups) stage(gi, 0, 0);
    if (!early) pdl_wait();
    x = after_wait(x);
    pdl_trigger();
    for (int it = 0; gi < ngroups; ++it, gi += gridDim.x) {
        const int tile = gi * tb + tl;
        float a[NT] = {};
        for (int j = 0; j < n; ++j, ++sidx) {
            const int cur = nbuf > 1 ? (sidx & 1) : 0;
            const int gn = j + 1 < n ? gi : gi + gridDim.x, jn = j + 1 < n ? j + 1 : 0;
            if (nbuf > 1 && gn < ngroups) {
                stage(gn, jn, cur ^ 1);  // that buffer was last read in the previous step, closed by its __syncwarp
                cp_wait<1>();
            } else {
                cp_wait<0>();
            }
            __syncwarp();
            if (tile < ntiles) {
                const uint8_t * rp = buf + cur * UB + (lane & 15) * CB;
#pragma unroll 1
                for (int q = 0; q < GPH; ++q) {
                    const int g = (lane >> 4) * GPH + q;
                    if (g >= GPC) break;
                    const float * xg = x + (c0 + j) * CV + g * 32;
                    float c[NT][4] = {};
                    DQ<TY>::run(rp, g, [&](int jj, float v0, float v1, float v2, float v3) {
                        fma4n<NT>(c, xg + jj, xs, nt, v0, v1, v2, v3);
                    }, tab);
#pragma unroll
                    for (int t = 0; t < NT; ++t) a[t] = __fadd_rn(a[t], chains(c[t]));
                }
            }
            __syncwarp();
            if (nbuf == 1 && gn < ngroups) stage(gn, jn, 0);
        }
        tile_store<NT>(a, red + (it & 1) * nw * 16 * NT, warp, lane, tl, wk, wpt, tile, ntiles, rows, nt, ks, y, ys, acc,
                       part);
    }
    if (early) pdl_wait();
}

// BF16 (CV 64, 128-byte rows): rows are 16-byte aligned, so each lane loads its 64 bytes straight into registers
// (the next step prefetched) instead of through shared memory, where 128-byte rows would all hit one bank.
// MIX: the hyper-connection up weight, finished by tile_mix (hn, zinj, xo, injo; y unused)
template <int NT, bool MIX>
__global__ void __launch_bounds__(256, 2) k_tgemv_bf16(const uint8_t * __restrict__ W, int ntiles, int nch, int rows, int wpt,
                                                       int ks, const float * __restrict__ x, int xs, float * __restrict__ y,
                                                       int ys, int nt, int acc, float * __restrict__ part, int early,
                                                       const float * __restrict__ hn, const float * __restrict__ zinj,
                                                       float * __restrict__ xo, float * __restrict__ injo) {
    extern __shared__ __align__(16) uint8_t sm[];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, nw = blockDim.x >> 5, tb = nw / wpt;
    const int tl = warp / wpt, wk = warp % wpt;
    const int S = wpt * ks, si = blockIdx.y * wpt + wk;
    const int c0 = si * nch / S, n = (si + 1) * nch / S - c0;
    const int ngroups = (ntiles + tb - 1) / tb;
    float * red = (float *) sm;
    const size_t lo = (size_t) (lane & 15) * 128 + (lane >> 4) * 64;
    auto load = [&](int gi, int j, uint4 (&q)[4]) {
        const int tile = gi * tb + tl;
        if (tile < ntiles) {
            const uint4 * src = (const uint4 *) (W + ((size_t) tile * nch + c0 + j) * 2048 + lo);
#pragma unroll
            for (int k = 0; k < 4; ++k) q[k] = __ldg(src + k);
        }
    };
    uint4 q[4] = {}, qn[4] = {};
    int gi = blockIdx.x;
    if (gi < ngroups) load(gi, 0, q);
    if (!early) pdl_wait();
    x = after_wait(x);
    if (MIX) {
        hn = after_wait(hn);
        zinj = after_wait(zinj);
    }
    pdl_trigger();
    for (int it = 0; gi < ngroups; ++it, gi += gridDim.x) {
        const int tile = gi * tb + tl;
        float a[NT] = {};
        [[maybe_unused]] float hv[NT] = {};
        if constexpr (MIX)
            if (wk == 0) mix_prefetch<NT>(hv, hn, tile, ntiles, lane, nt);
        for (int j = 0; j < n; ++j) {
            const int gn = j + 1 < n ? gi : gi + gridDim.x, jn = j + 1 < n ? j + 1 : 0;
            if (gn < ngroups) load(gn, jn, qn);
            if (tile < ntiles) {
                const float * xg = x + (c0 + j) * 64 + (lane >> 4) * 32;
                float c[NT][4] = {};
#pragma unroll
                for (int k = 0; k < 4; ++k) {
                    const float v0 = __uint_as_float(q[k].x << 16), v1 = __uint_as_float(q[k].x & 0xffff0000u);
                    const float v2 = __uint_as_float(q[k].y << 16), v3 = __uint_as_float(q[k].y & 0xffff0000u);
                    const float v4 = __uint_as_float(q[k].z << 16), v5 = __uint_as_float(q[k].z & 0xffff0000u);
                    const float v6 = __uint_as_float(q[k].w << 16), v7 = __uint_as_float(q[k].w & 0xffff0000u);
                    fma4n<NT>(c, xg + 8 * k, xs, nt, v0, v1, v2, v3);
                    fma4n<NT>(c, xg + 8 * k + 4, xs, nt, v4, v5, v6, v7);
                }
#pragma unroll
                for (int t = 0; t < NT; ++t) a[t] = __fadd_rn(a[t], chains(c[t]));
            }
#pragma unroll
            for (int k = 0; k < 4; ++k) q[k] = qn[k];
        }
        if constexpr (MIX)
            tile_mix<NT>(a, hv, red + (it & 1) * nw * 16 * NT, warp, lane, tl, wk, wpt, tile, ntiles, nt, zinj, xo, injo);
        else
            tile_store<NT>(a, red + (it & 1) * nw * 16 * NT, warp, lane, tl, wk, wpt, tile, ntiles, rows, nt, ks, y, ys,
                           acc, part);
    }
    if (early) pdl_wait();
}

// split-K finish: y[t][r] (+)= sum over splits k = 0, 1, ... of part[k][t][r].   grid (ceil(rows / 256), nt)
__global__ void k_tg_reduce(const float * __restrict__ part, int ks, int rpad, int rows, float * __restrict__ y, int ys,
                            int acc) {
    pdl_wait();
    pdl_trigger();
    const int t = blockIdx.y, r = blockIdx.x * 256 + threadIdx.x;
    if (r >= rows) return;
    float s = part[(size_t) t * rpad + r];
    for (int k = 1; k < ks; ++k) s += part[((size_t) k * MAXT + t) * rpad + r];
    float * o = y + (size_t) t * ys + r;
    *o = acc ? *o + s : s;
}

// routed experts, gate and up: act[pair][r] = silu(gate_e[r] . x[t]) * (up_e[r] . x[t]), e = sel[pair], t = pair / NEU.
// grid (ceil(40 / tb), npairs), tb * wpt warps: warp (tl, wk) owns expert tile blockIdx.x * tb + tl, chunk split wk of 10,
// and stages the gate and up units of each chunk together.
template <uint32_t TY>
__global__ void __launch_bounds__(256, 2) k_tmoe_gu(const uint8_t * __restrict__ Wg, const uint8_t * __restrict__ Wu,
                                                 const int * __restrict__ sel, const float * __restrict__ x,
                                                 float * __restrict__ act, int wpt) {
    constexpr int UB = 16 * qcb<TY, 256>(), NCH = D / 256, TPE = FFE / 16;
    extern __shared__ __align__(16) uint8_t sm[];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, nw = blockDim.x >> 5;
    const int tl = warp / wpt, wk = warp % wpt, te = blockIdx.x * (nw / wpt) + tl;
    const int pair = blockIdx.y, t = pair / NEU;
    const int nbuf = (NCH + wpt - 1) / wpt > 1 ? 2 : 1;
    const uint32_t * tab = stage_grid<TY>((uint32_t *) (sm + (size_t) nw * nbuf * 2 * UB + (size_t) nw * 16 * 2 * 4));
    pdl_wait();  // sel comes from k_route, the kernel just before
    const int e = after_wait(sel)[pair];
    const int c0 = wk * NCH / wpt, n = te < TPE ? (wk + 1) * NCH / wpt - c0 : 0;
    const size_t u0 = ((size_t) (e * TPE + te) * NCH + c0) * UB;
    const uint8_t * src[2] = {Wg + u0, Wu + u0};
    float a[2][1] = {};
    tile_dot<TY, 256, 2, 1>(src, n, sm + (size_t) warp * nbuf * 2 * UB, nbuf, x + (size_t) t * D + c0 * 256, 0, 1, a, lane, tab);
    float g = a[0][0] + __shfl_xor_sync(0xffffffffu, a[0][0], 16);
    float u = a[1][0] + __shfl_xor_sync(0xffffffffu, a[1][0], 16);
    if (wpt > 1) {
        float * red = (float *) (sm + (size_t) nw * nbuf * 2 * UB);
        if (lane < 16) { red[(warp * 16 + lane) * 2] = g; red[(warp * 16 + lane) * 2 + 1] = u; }
        __syncthreads();
        if (wk != 0 || lane >= 16) return;
        g = red[(tl * wpt * 16 + lane) * 2];
        u = red[(tl * wpt * 16 + lane) * 2 + 1];
        for (int k = 1; k < wpt; ++k) {
            g += red[((tl * wpt + k) * 16 + lane) * 2];
            u += red[((tl * wpt + k) * 16 + lane) * 2 + 1];
        }
    }
    if (lane < 16 && te < TPE) act[(size_t) pair * FFE + te * 16 + lane] = silu(g) * u;
}

// The same arithmetic on 8-row half tiles: 800 blocks of wpt warps instead of 400 spread 3 / 2 over the SMs (the busiest
// SM held 1.28x the average), and each lane decodes one matrix. Lane = (row 0..7, matrix gate/up, half of the unit's
// groups); per lane the sequence of FMAs, the half merge (shfl_xor 16) and the warp merge (wk order) are the ones of
// k_tmoe_gu, so the results are bit-identical.
template <uint32_t TY>
__global__ void __launch_bounds__(160, 5) k_tmoe_gu8(const uint8_t * __restrict__ Wg, const uint8_t * __restrict__ Wu,
                                                  const int * __restrict__ sel, const float * __restrict__ x,
                                                  float * __restrict__ act, int wpt) {
    constexpr int CB = qcb<TY, 256>(), HB = 8 * CB, NCH = D / 256, TPE = FFE / 8;
    extern __shared__ __align__(16) uint8_t sm[];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, nw = blockDim.x >> 5;
    const int tl = warp / wpt, wk = warp % wpt, te = blockIdx.x * (nw / wpt) + tl;  // 8-row tile of the expert
    const int pair = blockIdx.y, t = pair / NEU;
    const int nbuf = (NCH + wpt - 1) / wpt > 1 ? 2 : 1;
    const int r8 = lane & 7, m = (lane >> 3) & 1, h = lane >> 4;
    uint8_t * buf = sm + (size_t) warp * nbuf * 2 * HB;
    float * red = (float *) (sm + (size_t) nw * nbuf * 2 * HB);
    const uint32_t * tab = stage_grid<TY>((uint32_t *) (red + nw * 16));
    pdl_wait();  // sel comes from k_route, the kernel just before
    const int e = after_wait(sel)[pair];
    x = after_wait(x);
    const int c0 = wk * NCH / wpt, n = te < TPE ? (wk + 1) * NCH / wpt - c0 : 0;
    // unit (tile te / 2, chunk) of expert e; this tile is the (te & 1)-th half of it
    const size_t u0 = ((size_t) (e * (TPE / 2) + (te >> 1)) * NCH + c0) * (16 * CB) + (size_t) (te & 1) * HB;
    const uint8_t * src[2] = {Wg + u0, Wu + u0};
    auto stage = [&](int j, int sb) {
#pragma unroll
        for (int mm = 0; mm < 2; ++mm)
            for (int p = lane; p < HB / 16; p += 32) cp16(buf + (sb * 2 + mm) * HB + 16 * p, src[mm] + (size_t) j * (16 * CB) + 16 * p);
        cp_commit();
    };
    if (n > 0) stage(0, 0);
    pdl_trigger();
    const float * xt = x + (size_t) t * D + c0 * 256;
    float a = 0.0f;
    for (int j = 0; j < n; ++j) {
        const int cur = nbuf > 1 ? (j & 1) : 0;
        if (nbuf > 1 && j + 1 < n) {
            stage(j + 1, cur ^ 1);
            cp_wait<1>();
        } else {
            cp_wait<0>();
        }
        __syncwarp();
        const uint8_t * rp = buf + (cur * 2 + m) * HB + r8 * CB;
#pragma unroll 1
        for (int q = 0; q < 4; ++q) {
            const int g = h * 4 + q;
            const float * xg = xt + j * 256 + g * 32;
            float c[1][4] = {};
            DQ<TY>::run(rp, g, [&](int jj, float v0, float v1, float v2, float v3) {
                fma4n<1>(c, xg + jj, 0, 1, v0, v1, v2, v3);
            }, tab);
            a = __fadd_rn(a, chains(c[0]));
        }
        __syncwarp();
        if (nbuf == 1 && j + 1 < n) stage(j + 1, 0);
    }
    a += __shfl_xor_sync(0xffffffffu, a, 16);
    float g = a, u = 0.0f;
    if (wpt > 1) {
        if (lane < 16) red[warp * 16 + lane] = a;  // lane = m * 8 + r8
        __syncthreads();
        if (wk != 0 || lane >= 8) return;
        g = red[tl * wpt * 16 + lane];
        u = red[tl * wpt * 16 + 8 + lane];
        for (int k = 1; k < wpt; ++k) {
            g += red[(tl * wpt + k) * 16 + lane];
            u += red[(tl * wpt + k) * 16 + 8 + lane];
        }
    } else {
        u = __shfl_sync(0xffffffffu, a, lane + 8);
    }
    if (lane < 8 && te < TPE) act[(size_t) pair * FFE + te * 8 + lane] = silu(g) * u;
}

// routed experts, down: dpart[pair][r] = down_e[r] . act[pair] (the slot weights are applied in k_hc_add).
// grid (ceil(160 / tb), npairs), tb * wpt warps; 640 columns = 5 chunks of 128.
template <uint32_t TY>
__global__ void __launch_bounds__(256, 2) k_tmoe_down(const uint8_t * __restrict__ Wd, const int * __restrict__ sel,
                                                   const float * __restrict__ act, float * __restrict__ dpart, int wpt) {
    constexpr int UB = 16 * qcb<TY, 128>(), NCH = FFE / 128, TPE = D / 16;
    extern __shared__ __align__(16) uint8_t sm[];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, nw = blockDim.x >> 5;
    const int tl = warp / wpt, wk = warp % wpt, te = blockIdx.x * (nw / wpt) + tl;
    const int pair = blockIdx.y, e = sel[pair];
    const int nbuf = (NCH + wpt - 1) / wpt > 1 ? 2 : 1;
    const int c0 = wk * NCH / wpt, n = te < TPE ? (wk + 1) * NCH / wpt - c0 : 0;
    const uint8_t * src = Wd + ((size_t) (e * TPE + te) * NCH + c0) * UB;
    float a[1][1] = {};
    tile_dot<TY, 128, 1, 1>(&src, n, sm + (size_t) warp * nbuf * UB, nbuf, act + (size_t) pair * FFE + c0 * 128, 0, 1, a,
                         lane, nullptr);
    float s = a[0][0] + __shfl_xor_sync(0xffffffffu, a[0][0], 16);
    if (wpt > 1) {
        float * red = (float *) (sm + (size_t) nw * nbuf * UB);
        if (lane < 16) red[warp * 16 + lane] = s;
        __syncthreads();
        if (wk != 0 || lane >= 16) return;
        s = red[tl * wpt * 16 + lane];
        for (int k = 1; k < wpt; ++k) s += red[(tl * wpt + k) * 16 + lane];
    }
    if (lane < 16 && te < TPE) dpart[(size_t) pair * D + te * 16 + lane] = s;
}

// row i of tile T of a hyper-connection up weight (rows s * D + c for stream s, column c) in the order tile_mix reads:
// tile T holds columns 4T..4T+3, row 4 * (c % 4) + s
__host__ __device__ inline uint64_t hc_perm_row(uint64_t T, int i) { return (uint64_t) (i & 3) * D + T * 4 + (i >> 2); }

// repack GGUF rows into tiled units (see above); hcperm: rows in hc_perm_row order.  one warp per unit, 8 per block
__global__ void k_repack(const uint8_t * __restrict__ src, uint8_t * __restrict__ dst, int rows, uint64_t rb, int cb,
                         int nch, uint64_t nunits, int hcperm) {
    const uint64_t u = blockIdx.x * 8ull + (threadIdx.x >> 5);
    if (u >= nunits) return;
    const int lane = threadIdx.x & 31;
    const uint64_t T = u / nch, c = u % nch;
    uint8_t * d = dst + u * 16 * cb;
    for (int i = 0; i < 16; ++i) {
        const uint64_t r = hcperm ? hc_perm_row(T, i) : T * 16 + i;
        const uint8_t * s = src + r * rb + c * cb;
        for (int b = lane; b < cb; b += 32) d[i * cb + b] = r < (uint64_t) rows ? s[b] : 0;
    }
}

// dequantize whole rows (unit tests and embedding lookups): out[r][c]
template <uint32_t TY>
__global__ void k_dequant_rows(const uint8_t * __restrict__ W, uint64_t rb, const int * __restrict__ ids, int ngroups,
                               float * __restrict__ out) {
    const int r = blockIdx.x;
    const uint8_t * row = W + (size_t) (ids ? ids[r] : r) * rb;
    for (int g = threadIdx.x; g < ngroups; g += blockDim.x) {
        float v[32];
        dq32<TY>(row, g, v);
#pragma unroll
        for (int j = 0; j < 32; ++j) out[(size_t) r * ngroups * 32 + g * 32 + j] = v[j];
    }
}

// ---------------------------------------------------------------------------------------------
// elementwise / normalization kernels

// out[t][s*D + c] = h[t][s*D + c] * rsqrt(mean_c h^2 + eps) * w[s*D + c]  (RMSNorm per stream, w stored as 1+w)
// grid (nt, ngroups), D / 4 threads (four values each, all loads issued first); group width D
__global__ void __launch_bounds__(D / 4) k_groupnorm(const float * __restrict__ h, const float * __restrict__ w,
                                                     float * __restrict__ out, int stride) {
    pdl_wait();
    __shared__ float red[D / 128];
    const size_t base = (size_t) blockIdx.x * stride + blockIdx.y * D;
    const float4 v = ((const float4 *) (h + base))[threadIdx.x];
    const float4 wv = ((const float4 *) (w + blockIdx.y * D))[threadIdx.x];
    const float ss = block_sum<D / 128>(v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w, red);
    pdl_trigger();
    const float r = rsqrtf(ss / D + EPS);
    ((float4 *) (out + base))[threadIdx.x] = make_float4(v.x * r * wv.x, v.y * r * wv.y, v.z * r * wv.z, v.w * r * wv.w);
}

// A GEMV result as its consumer reads it: value(t, r) = sum over splits k = 0, 1, ... of p[k * kst + t * ts + r].
// With ks == 1 it is the output buffer itself; with ks > 1 the consumer adds the split-K partials, in the same order
// k_tg_reduce would, so folding the reduction into the consumer changes no bit.
struct GOut {
    const float * p;
    int ks, ts;
    size_t kst;
};
__device__ __forceinline__ float gval(const GOut & o, int t, int r) {
    const float * q = o.p + (size_t) t * o.ts + r;
    float s = q[0];
    for (int k = 1; k < o.ks; ++k) s += q[k * o.kst];
    return s;
}

// gval of the four values c..c+3 (c a multiple of 4): float4 loads of every split issued first, then added in split
// order like gval
constexpr int GV_MAXK = 8;
__device__ __forceinline__ float4 gval4(const GOut & o, int t, int c) {
    const float * q = o.p + (size_t) t * o.ts + c;
    if (o.ks > GV_MAXK) return make_float4(gval(o, t, c), gval(o, t, c + 1), gval(o, t, c + 2), gval(o, t, c + 3));
    float4 v[GV_MAXK];
#pragma unroll
    for (int k = 0; k < GV_MAXK; ++k)
        if (k < o.ks) v[k] = *(const float4 *) (q + k * o.kst);
    float4 s = v[0];
#pragma unroll
    for (int k = 1; k < GV_MAXK; ++k)
        if (k < o.ks) { s.x += v[k].x; s.y += v[k].y; s.z += v[k].z; s.w += v[k].w; }
    return s;
}

// hyper-connection low rank: t320[t][r] = silu(dn[t][r] / HC) for r < HCR; zinj[t][s] = dn[t][HCR + s] (rows of the
// stacked down+inject weight; none for the final mixer).   grid nt, 384 threads
__global__ void k_hc_lowrank(const GOut dn, float * __restrict__ t320, float * __restrict__ zinj, int has_inj) {
    pdl_wait();
    const int t = blockIdx.x, r = threadIdx.x;
    const float v = r < HCR || (has_inj && r < HCR + HC) ? gval(dn, t, r) : 0.0f;
    pdl_trigger();
    if (r < HCR) t320[t * HCR + r] = silu(v * (1.0f / HC));
    else if (has_inj && r < HCR + HC) zinj[t * HC + r - HCR] = v;
}

// hyper-connection write h[t][s][c] += y[t][c] * inj[t][s], then (norm non-null) the RMSNorm of the stream for the
// next hyper-connection read: hn[t][s][c] = h * rsqrt(mean_c h^2 + eps) * norm[s][c], with k_groupnorm's thread layout
// and order. y is a GEMV result or, with dpart non-null, the MoE output sum_k wsel[t][k] dpart[t * NEU + k][c] +
// sigmoid(sg[t]) shared[t][c] (sg = row NE of the router + shared-gate result rsg; shared = y), formed here.
// grid (nt, HC), D / 4 threads (four values each)
__global__ void __launch_bounds__(D / 4) k_hc_add(float * __restrict__ h, const GOut y, const float * __restrict__ inj,
                                                  const float * __restrict__ dpart, const float * __restrict__ wsel,
                                                  const GOut rsg, const float * __restrict__ norm, float * __restrict__ hn) {
    pdl_wait();
    inj = after_wait(inj);
    dpart = after_wait(dpart);
    wsel = after_wait(wsel);
    __shared__ float red[D / 128];
    const int t = blockIdx.x, s = blockIdx.y, c0 = threadIdx.x * 4;
    float4 * hp = (float4 *) (h + (size_t) t * HCD + s * D) + threadIdx.x;
    float4 v = *hp;
    const float is = inj[t * HC + s];
    const float4 g = gval4(y, t, c0);
    float yv[4] = {g.x, g.y, g.z, g.w};
    if (dpart) {
        float4 dp[NEU];
        float wk[NEU];
#pragma unroll
        for (int k = 0; k < NEU; ++k) {
            dp[k] = *(const float4 *) (dpart + (size_t) (t * NEU + k) * D + c0);
            wk[k] = wsel[t * NEU + k];
        }
        const float sg = sigm(gval(rsg, t, NE));
        float m[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
        for (int k = 0; k < NEU; ++k) {
            m[0] += wk[k] * dp[k].x;
            m[1] += wk[k] * dp[k].y;
            m[2] += wk[k] * dp[k].z;
            m[3] += wk[k] * dp[k].w;
        }
#pragma unroll
        for (int j = 0; j < 4; ++j) yv[j] = m[j] + sg * yv[j];
    }
    pdl_trigger();
    v.x += yv[0] * is;
    v.y += yv[1] * is;
    v.z += yv[2] * is;
    v.w += yv[3] * is;
    *hp = v;
    if (!norm) return;
    const float4 wv = ((const float4 *) (norm + s * D))[threadIdx.x];
    const float ss = block_sum<D / 128>(dot4c(v.x, v.x, v.y, v.y, v.z, v.z, v.w, v.w), red);
    const float r = rsqrtf(ss / D + EPS);
    ((float4 *) (hn + (size_t) t * HCD + s * D))[threadIdx.x] =
        make_float4(v.x * r * wv.x, v.y * r * wv.y, v.z * r * wv.z, v.w * r * wv.w);
}

// token embedding (IQ3_S rows) copied into all HC streams.   grid nt, 128 threads
__global__ void k_embed(const uint8_t * __restrict__ tok, uint64_t rb, const int * __restrict__ ids, float * __restrict__ h) {
    pdl_wait();
    pdl_trigger();
    const int t = blockIdx.x;
    const uint8_t * row = tok + (size_t) ids[t] * rb;
    for (int g = threadIdx.x; g < D / 32; g += blockDim.x) {
        float v[32];
        dq32<T_IQ3_S>(row, g, v);
#pragma unroll
        for (int k = 0; k < HC; ++k)
#pragma unroll
            for (int j = 0; j < 32; ++j) h[(size_t) t * HCD + k * D + g * 32 + j] = v[j];
    }
}

// ---------------------------------------------------------------------------------------------
// gated delta net (linear attention), plain advancing state

constexpr float GDN_SCALE = 0.08838834764831845f;  // 1/sqrt(128)
constexpr int GDN_CST = 3 * DINNER + NVH * 3 * 2 * DS;  // conv state floats per layer (see k_gdn)

// The DeltaNet core of one value head h (key head kh = h % NKH) for nt tokens, in one kernel:
//  conv: causal depthwise conv (kernel 4) over [state(3), x], silu, for the head's q, k (channels of key head kh) and
//        v channels; state advanced. The value channels' state is [3][DINNER]; every value head keeps its own copy of
//        its key head's q/k state ([NVH][3][2][DS]: the three value heads of a key head would otherwise race on it),
//        and the copies stay equal (same inputs, same arithmetic).
//  l2-normalize q and k: x * rsqrt(sum x^2 + 1e-6)
//  gates: beta = sigmoid(b), g = ssm_a * softplus(a + dt_bias) (ssm_a = -exp(A_log)); ab = rows [alpha; beta]
//  delta rule: S = e^g S + k (beta (v - e^g S^T k))^T, o = S^T q / sqrt(128); state advanced
//  gated RMSNorm: on = o * rsqrt(mean o^2 + eps) * w * sigmoid(z)
// Each step is the arithmetic of the separate kernel it replaces, in the same thread layout (thread = dimension), so
// the results are the same bits.   grid NVH, DS threads; thread col holds column col of S
__global__ void __launch_bounds__(DS) k_gdn(const float * __restrict__ qkv, float * __restrict__ cst,
                                            const float * __restrict__ cw, const GOut ab, const float * __restrict__ dt_bias,
                                            const float * __restrict__ ssm_a, float * __restrict__ S,
                                            const float * __restrict__ z, const float * __restrict__ nw,
                                            float * __restrict__ on, int nt) {
    pdl_wait();
    qkv = after_wait(qkv);
    z = after_wait(z);
    const int h = blockIdx.x, col = threadIdx.x, kh = h % NKH;
    __shared__ float red[4];
    __shared__ float ks[DS], qs[DS];
    const int cq = kh * DS + col, ck = NKH * DS + kh * DS + col, cv = 2 * NKH * DS + h * DS + col;
    float * stv = cst + h * DS + col;                              // [3][DINNER]
    float * stq = cst + 3 * DINNER + (size_t) h * 3 * 2 * DS + col;  // [3][2][DS] of this value head
    float hq0 = stq[0], hq1 = stq[2 * DS], hq2 = stq[4 * DS];
    float hk0 = stq[DS], hk1 = stq[3 * DS], hk2 = stq[5 * DS];
    float hv0 = stv[0], hv1 = stv[DINNER], hv2 = stv[2 * DINNER];
    const float4 wq = ((const float4 *) cw)[cq], wk = ((const float4 *) cw)[ck], wv = ((const float4 *) cw)[cv];
    const float nwc = nw[col], dtb = dt_bias[h], sa = ssm_a[h];
    float s[DS];
    float * Sh = S + (size_t) h * DS * DS;
#pragma unroll
    for (int i = 0; i < DS; ++i) s[i] = Sh[i * DS + col];
    pdl_trigger();
    for (int t = 0; t < nt; ++t) {
        const float * row = qkv + (size_t) t * CONVCH;
        const float xq = row[cq], xk = row[ck], xv = row[cv];
        float q = silu(dot4c(hq0, wq.x, hq1, wq.y, hq2, wq.z, xq, wq.w));
        float k = silu(dot4c(hk0, wk.x, hk1, wk.y, hk2, wk.z, xk, wk.w));
        const float vv = silu(dot4c(hv0, wv.x, hv1, wv.y, hv2, wv.z, xv, wv.w));
        hq0 = hq1; hq1 = hq2; hq2 = xq;
        hk0 = hk1; hk1 = hk2; hk2 = xk;
        hv0 = hv1; hv1 = hv2; hv2 = xv;
        const float ssq = block_sum<4>(q * q, red);
        q = q * rsqrtf(ssq + 1e-6f);
        const float ssk = block_sum<4>(k * k, red);
        k = k * rsqrtf(ssk + 1e-6f);
        __syncthreads();
        qs[col] = q;
        ks[col] = k;
        __syncthreads();
        const float bv = sigm(gval(ab, t, NVH + h));
        const float a = gval(ab, t, h) + dtb;
        const float sp = a > 20.0f ? a : log1pf(expf(a));
        const float gv = expf(sp * sa);
        float kv = __fmul_rn(s[0], ks[0]);
#pragma unroll
        for (int i = 1; i < DS; ++i) kv = __fmaf_rn(s[i], ks[i], kv);
        const float delta = __fmul_rn(__fmaf_rn(-gv, kv, vv), bv);
        float o = 0.0f;
#pragma unroll
        for (int i = 0; i < DS; ++i) {
            s[i] = __fmaf_rn(gv, s[i], __fmul_rn(ks[i], delta));
            o = i ? __fmaf_rn(s[i], qs[i], o) : __fmul_rn(s[i], qs[i]);
        }
        o = o * GDN_SCALE;
        const float ss = block_sum<4>(o * o, red);
        const size_t oi = (size_t) t * DINNER + h * DS + col;
        on[oi] = o * rsqrtf(ss / DS + EPS) * nwc * sigm(z[oi]);
    }
#pragma unroll
    for (int i = 0; i < DS; ++i) Sh[i * DS + col] = s[i];
    stq[0] = hq0; stq[2 * DS] = hq1; stq[4 * DS] = hq2;
    stq[DS] = hk0; stq[3 * DS] = hk1; stq[5 * DS] = hk2;
    stv[0] = hv0; stv[DINNER] = hv1; stv[2 * DINNER] = hv2;
}

// ---------------------------------------------------------------------------------------------
// attention with the QSA indexer

__constant__ float c_inv_freq[NROT / 2];

// rotate the first NROT dims of x (pairs i, i + NROT/2) at position pos
__device__ __forceinline__ void rope_pair(float & x0, float & x1, int i, int pos) {
    const float ang = (float) pos * c_inv_freq[i];
    const float cs = cosf(ang), sn = sinf(ang);
    const float a = x0, b = x1;
    x0 = a * cs - b * sn;
    x1 = b * cs + a * sn;
}

// Positions live in device memory (d_pos = tokens already in the cache) so a forward can be replayed as a CUDA graph.

// q heads: Q = rope(rmsnorm(q) * wq); gate kept in qg. k: rope(rmsnorm(k) * wk) -> K cache; v -> V cache.
// grid (nt, NH + 2 * NKV), 256 threads (one per head dim)
__global__ void k_qk_norm_rope(const float * __restrict__ qg, const GOut kk, const GOut vv,
                               const float * __restrict__ wq, const float * __restrict__ wk, float * __restrict__ Q,
                               float * __restrict__ Kc, float * __restrict__ Vc, const int * __restrict__ d_pos) {
    pdl_wait();
    pdl_trigger();
    __shared__ float red[8];
    __shared__ float buf[HD];
    const int t = blockIdx.x, j = blockIdx.y, d = threadIdx.x, pos = *d_pos + t;
    if (j >= NH + NKV) {  // values
        const int kv = j - NH - NKV;
        Vc[((size_t) pos * NKV + kv) * HD + d] = gval(vv, t, kv * HD + d);
        return;
    }
    const bool isq = j < NH;
    const float x = isq ? qg[(size_t) t * NH * QG + j * QG + d] : gval(kk, t, (j - NH) * HD + d);
    const float ss = block_sum<8>(x * x, red);
    buf[d] = x * rsqrtf(ss / HD + EPS) * (isq ? wq[d] : wk[d]);
    __syncthreads();
    float y = buf[d];
    if (d < NROT) {
        const int i = d % (NROT / 2);
        float x0 = buf[i], x1 = buf[i + NROT / 2];
        rope_pair(x0, x1, i, pos);
        y = d < NROT / 2 ? x0 : x1;
    }
    if (isq) Q[((size_t) t * NH + j) * HD + d] = y;
    else Kc[((size_t) pos * NKV + (j - NH)) * HD + d] = y;
}

// indexer: Qix = rope(rmsnorm(iq) * wqn) per head; raw index keys -> IKc[pos].   grid (nt, IXH + 1), 128 threads
// iqk = rows [indexer q (IXH heads); indexer k]
__global__ void k_ix_qk(const GOut iqk, const float * __restrict__ wqn,
                        float * __restrict__ Qix, float * __restrict__ IKc, const int * __restrict__ d_pos) {
    pdl_wait();
    pdl_trigger();
    __shared__ float red[4];
    __shared__ float buf[IXD];
    const int t = blockIdx.x, j = blockIdx.y, d = threadIdx.x, pos = *d_pos + t;
    if (j == IXH) {
        IKc[(size_t) pos * IXD + d] = gval(iqk, t, IXH * IXD + d);
        return;
    }
    const float x = gval(iqk, t, j * IXD + d);
    const float ss = block_sum<4>(x * x, red);
    buf[d] = x * rsqrtf(ss / IXD + EPS) * wqn[d];
    __syncthreads();
    float y = buf[d];
    if (d < NROT) {
        const int i = d % (NROT / 2);
        float x0 = buf[i], x1 = buf[i + NROT / 2];
        rope_pair(x0, x1, i, pos);
        y = d < NROT / 2 ? x0 : x1;
    }
    Qix[((size_t) t * IXH + j) * IXD + d] = y;
}

// block keys for the blocks this forward completes: rope(rmsnorm(mean of 4 raw keys) * wkn) at position 4b.
// grid nt / IXR + 1, 128 threads
__global__ void k_ix_blocks(const float * __restrict__ IKc, const float * __restrict__ wkn, float * __restrict__ BKc,
                            const int * __restrict__ d_pos, int nt) {
    pdl_wait();
    pdl_trigger();
    __shared__ float red[4];
    __shared__ float buf[IXD];
    const int b = *d_pos / IXR + blockIdx.x, d = threadIdx.x;
    if (b >= (*d_pos + nt) / IXR) return;
    const float * k = IKc + (size_t) b * IXR * IXD;
    const float x = (k[d] + k[IXD + d] + k[2 * IXD + d] + k[3 * IXD + d]) / IXR;
    const float ss = block_sum<4>(x * x, red);
    buf[d] = x * rsqrtf(ss / IXD + EPS) * wkn[d];
    __syncthreads();
    float y = buf[d];
    if (d < NROT) {
        const int i = d % (NROT / 2);
        float x0 = buf[i], x1 = buf[i + NROT / 2];
        rope_pair(x0, x1, i, b * IXR);
        y = d < NROT / 2 ? x0 : x1;
    }
    BKc[(size_t) b * IXD + d] = y;
}

// block scores for query t: sum_h relu(Qix[t][h] . BK[b]) / sqrt(IXD), b < nblocks(t) = (pos+1)/4; only computed
// when the indexer is active (nblocks > IXTOP).   grid (nt, 64), 256 threads: one warp per block, strided
__global__ void k_ix_score(const float * __restrict__ Qix, const float * __restrict__ BKc, float * __restrict__ S, int sstride,
                           const int * __restrict__ d_pos) {
    pdl_wait();
    pdl_trigger();
    const int t = blockIdx.x, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int nb = (*d_pos + t + 1) / IXR;
    if (nb <= IXTOP) return;
    for (int b = blockIdx.y * 8 + warp; b < nb; b += gridDim.y * 8) {
        const float * k = BKc + (size_t) b * IXD;
        float s = 0.0f;
#pragma unroll
        for (int h = 0; h < IXH; ++h) {
            const float * q = Qix + ((size_t) t * IXH + h) * IXD;
            float p = 0.0f;
#pragma unroll
            for (int i = lane; i < IXD; i += 32) p += q[i] * k[i];
            p = warp_sum(p);
            s += fmaxf(p, 0.0f);
        }
        if (lane == 0) S[(size_t) t * sstride + b] = s / sqrtf((float) IXD);
    }
}

// keep the IXTOP highest-scoring blocks of each query (scores are >= 0, so their float bits order like the values).
// mask[t][b] = 1 for kept blocks; ties at the threshold are kept lowest index first.   grid nt, 1024 threads
__global__ void k_ix_select(const float * __restrict__ S, int sstride, uint8_t * __restrict__ mask, const int * __restrict__ d_pos) {
    pdl_wait();
    pdl_trigger();
    __shared__ int cnt;
    __shared__ int red[32];
    const int t = blockIdx.x, nb = (*d_pos + t + 1) / IXR;
    const uint32_t * s = (const uint32_t *) (S + (size_t) t * sstride);
    uint8_t * m = mask + (size_t) t * sstride;
    if (nb <= IXTOP) return;
    // largest threshold thr with count(s >= thr) >= IXTOP, by bitwise binary search
    uint32_t thr = 0;
    for (int bit = 31; bit >= 0; --bit) {
        const uint32_t cand = thr | (1u << bit);
        int c = 0;
        for (int b = threadIdx.x; b < nb; b += blockDim.x) c += s[b] >= cand;
        if (threadIdx.x == 0) cnt = 0;
        __syncthreads();
        atomicAdd(&cnt, c);
        __syncthreads();
        if (cnt >= IXTOP) thr = cand;
        __syncthreads();
    }
    // above threshold: kept; equal: kept in index order until IXTOP
    int above = 0;
    for (int b = threadIdx.x; b < nb; b += blockDim.x) above += s[b] > thr;
    if (threadIdx.x == 0) cnt = 0;
    __syncthreads();
    atomicAdd(&cnt, above);
    __syncthreads();
    int room = IXTOP - cnt;
    __syncthreads();
    for (int base = 0; base < nb; base += blockDim.x) {
        const int b = base + threadIdx.x;
        const bool eq = b < nb && s[b] == thr;
        // exclusive prefix count of equal entries within this chunk
        const unsigned ball = __ballot_sync(0xffffffffu, eq);
        const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
        if (lane == 0) red[w] = __popc(ball);
        __syncthreads();
        int before = __popc(ball & ((1u << lane) - 1));
        for (int i = 0; i < w; ++i) before += red[i];
        int total = 0;
        for (int i = 0; i < 32; ++i) total += red[i];
        if (b < nb) m[b] = (s[b] > thr) || (eq && before < room);
        __syncthreads();
        room -= total;
        if (room < 0) room = 0;
    }
}

// split-KV attention for query t, head h (kv head h / (NH/NKV)): split sp owns an equal share of keys 0..pos and walks
// it in tiles of ATT_CH with an online softmax. With the indexer active (nblocks > IXTOP) a key is visible when its
// block is kept or it is in the incomplete tail.   grid (nt, NH, NSPLIT), 256 threads (thread d owns output dim d)
__global__ void __launch_bounds__(256) k_attn_split(const float * __restrict__ Q, const float * __restrict__ Kc,
                                                    const float * __restrict__ Vc, const uint8_t * __restrict__ mask,
                                                    int mstride, float * __restrict__ part_o, float * __restrict__ part_ml,
                                                    const int * __restrict__ d_pos) {
    pdl_wait();
    pdl_trigger();
    __shared__ float q[HD];
    __shared__ float sc[ATT_CH];
    __shared__ float red[8];
    const int t = blockIdx.x, h = blockIdx.y, sp = blockIdx.z, pos = *d_pos + t, kv = h / (NH / NKV);
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, d = threadIdx.x;
    const int share = (pos + 1 + NSPLIT - 1) / NSPLIT;
    const int k0 = sp * share, k1 = min(k0 + share, pos + 1);
    const size_t pidx = ((size_t) t * NH + h) * NSPLIT + sp;
    float m = -INFINITY, l = 0.0f, o = 0.0f;
    q[d] = Q[((size_t) t * NH + h) * HD + d];
    const int nb = (pos + 1) / IXR;
    const bool sel = mask && nb > IXTOP;
    const uint8_t * mt = mask ? mask + (size_t) t * mstride : nullptr;
    __syncthreads();
    for (int tk = k0; tk < k1; tk += ATT_CH) {
        const int n = min(ATT_CH, k1 - tk);
        for (int i = warp; i < n; i += 8) {
            const int k = tk + i;
            const bool vis = !sel || k / IXR >= nb || mt[k / IXR];
            float s = -INFINITY;
            if (vis) {
                const float * kr = Kc + ((size_t) k * NKV + kv) * HD;
                float p = 0.0f;
#pragma unroll
                for (int dd = lane; dd < HD; dd += 32) p += q[dd] * kr[dd];
                s = warp_sum(p) * 0.0625f;  // 1/sqrt(256)
            }
            if (lane == 0) sc[i] = s;
        }
        __syncthreads();
        float mt_ = -INFINITY;
        for (int i = threadIdx.x; i < n; i += 256) mt_ = fmaxf(mt_, sc[i]);
        mt_ = block_max<8>(mt_, red);
        const float mn = fmaxf(m, mt_);
        if (mn != -INFINITY) {
            const float corr = m == -INFINITY ? 0.0f : expf(m - mn);
            float ls = 0.0f, os = 0.0f;
            for (int i = 0; i < n; ++i) {
                const float e = sc[i] == -INFINITY ? 0.0f : expf(sc[i] - mn);
                ls += e;
                os += e * Vc[((size_t) (tk + i) * NKV + kv) * HD + d];
            }
            l = l * corr + ls;
            o = o * corr + os;
            m = mn;
        }
        __syncthreads();
    }
    part_o[pidx * HD + d] = o;
    if (threadIdx.x == 0) { part_ml[pidx * 2] = m; part_ml[pidx * 2 + 1] = l; }
}

// combine splits, then the sigmoid output gate: og[t][h*HD + d] = attn * sigmoid(gate).   grid (nt, NH), 256 threads
__global__ void k_attn_combine(const float * __restrict__ part_o, const float * __restrict__ part_ml, const float * __restrict__ qg,
                               float * __restrict__ og) {
    pdl_wait();
    constexpr int nsplit = NSPLIT;
    const int t = blockIdx.x, h = blockIdx.y, d = threadIdx.x;
    const size_t base = ((size_t) t * NH + h) * nsplit;
    float M = -INFINITY;
    for (int s = 0; s < nsplit; ++s) M = fmaxf(M, part_ml[(base + s) * 2]);
    float L = 0.0f, o = 0.0f;
    for (int s = 0; s < nsplit; ++s) {
        const float m = part_ml[(base + s) * 2];
        if (m == -INFINITY) continue;
        const float w = expf(m - M);
        L += w * part_ml[(base + s) * 2 + 1];
        o += w * part_o[(base + s) * HD + d];
    }
    const float gate = qg[(size_t) t * NH * QG + h * QG + HD + d];
    pdl_trigger();
    og[(size_t) t * NH * HD + h * HD + d] = (o / L) * sigm(gate);
}

// ---------------------------------------------------------------------------------------------
// routing

// The routing as first written (one thread ranks each of the 160 candidates); kept as the reference fn_bench_route
// checks k_route against.   grid nt, 512 threads
__global__ void __launch_bounds__(NE) k_route_ref(const GOut logits, int * __restrict__ sel, float * __restrict__ wsel) {
    pdl_wait();
    __shared__ float red[NE / 32];
    __shared__ float cv[NE / 32 * NEU];
    __shared__ int ci[NE / 32 * NEU];
    __shared__ float topv[NEU];
    __shared__ int topi[NEU];
    const int t = blockIdx.x, i = threadIdx.x, lane = i & 31, w = i >> 5;
    const float l = gval(logits, t, i);
    const float mx = block_max<NE / 32>(l, red);
    pdl_trigger();
    const float v = expf(l - mx);
    int rk = 0;
#pragma unroll
    for (int o = 1; o < 32; ++o) {
        const int j = (lane + o) & 31;
        const float vj = __shfl_sync(0xffffffffu, v, j);
        rk += vj > v || (vj == v && j < lane);
    }
    if (rk < NEU) {
        cv[w * NEU + rk] = v;
        ci[w * NEU + rk] = i;
    }
    __syncthreads();
    if (i < NE / 32 * NEU) {
        const float c = cv[i];
        const int cx = ci[i];
        int r = 0;
        for (int j = 0; j < NE / 32 * NEU; ++j) r += cv[j] > c || (cv[j] == c && ci[j] < cx);
        if (r < NEU) {
            topv[r] = c;
            topi[r] = cx;
        }
    }
    __syncthreads();
    if (i == 0) {
        float s = 0.0f;
        for (int k = 0; k < NEU; ++k) s += topv[k];
        for (int k = 0; k < NEU; ++k) {
            sel[t * NEU + k] = topi[k];
            wsel[t * NEU + k] = topv[k] / s;
        }
    }
}

// softmax over 512 router logits (rows 0..511 of the router + shared-gate weight), top-10 in the order: probability
// descending, ties to the lower expert index (what repeated argmax picks), renormalized weights.
// Each warp ranks its 32 experts by shuffles and keeps its 10 best; each of the 160 candidates is ranked against the
// others by three threads (a third each, counts added). Also the shared expert's swiglu (a null: none):
// a[t][r] = silu(g[t][r]) * u[t][r].   grid nt, 512 threads
__global__ void __launch_bounds__(NE) k_route(const GOut logits, int * __restrict__ sel, float * __restrict__ wsel,
                                              const GOut g, const GOut u, float * __restrict__ a) {
    pdl_wait();
    constexpr int NC = NE / 32 * NEU, PART = (NC + 2) / 3;
    __shared__ float red[NE / 32];
    __shared__ float cv[NC];
    __shared__ int ci[NC];
    __shared__ int cnt[3][NC];
    __shared__ float topv[NEU];
    __shared__ int topi[NEU];
    const int t = blockIdx.x, i = threadIdx.x, lane = i & 31, w = i >> 5;
    if (a)
        for (int r = i; r < FFE; r += NE) a[t * FFE + r] = silu(gval(g, t, r)) * gval(u, t, r);
    const float l = gval(logits, t, i);
    const float mx = block_max<NE / 32>(l, red);
    pdl_trigger();
    const float v = expf(l - mx);
    int rk = 0;
#pragma unroll
    for (int o = 1; o < 32; ++o) {
        const int j = (lane + o) & 31;
        const float vj = __shfl_sync(0xffffffffu, v, j);
        rk += vj > v || (vj == v && j < lane);
    }
    if (rk < NEU) {
        cv[w * NEU + rk] = v;
        ci[w * NEU + rk] = i;
    }
    __syncthreads();
    if (i < 3 * NC) {
        const int k = i % NC, p = i / NC;
        const float c = cv[k];
        const int cx = ci[k];
        int r = 0;
#pragma unroll 6
        for (int j = p * PART; j < min(NC, (p + 1) * PART); ++j) r += cv[j] > c || (cv[j] == c && ci[j] < cx);
        cnt[p][k] = r;
    }
    __syncthreads();
    if (i < NC) {
        const int r = cnt[0][i] + cnt[1][i] + cnt[2][i];
        if (r < NEU) {
            topv[r] = cv[i];
            topi[r] = ci[i];
        }
    }
    __syncthreads();
    if (i < NEU) {
        float s = 0.0f;
        for (int k = 0; k < NEU; ++k) s += topv[k];
        sel[t * NEU + i] = topi[i];
        wsel[t * NEU + i] = topv[i] / s;
    }
}

// ---------------------------------------------------------------------------------------------
// per-layer n-gram embedding (layer 1)

__constant__ unsigned long long c_ple_mult[3];
__constant__ unsigned long long c_ple_size[PLE_NH];
__constant__ unsigned long long c_ple_off[PLE_NH];

// table rows of the 16 hashed n-gram heads for position pos + t (n-gram contexts never reach back across an EOS)
__global__ void k_ple_ids(const int * __restrict__ tok_all, const int * __restrict__ d_pos, int nt,
                          unsigned long long * __restrict__ rows) {
    pdl_wait();
    pdl_trigger();
    const int t = blockIdx.x, hd = threadIdx.x;
    if (t >= nt || hd >= PLE_NH) return;
    const int p = *d_pos + t;
    const unsigned long long t0 = (unsigned long long) tok_all[p];
    const int p1 = p >= 1 ? tok_all[p - 1] : PLE_EOS;
    const int p2 = (p >= 2 && p1 != PLE_EOS) ? tok_all[p - 2] : PLE_EOS;
    unsigned long long mixed = (t0 * c_ple_mult[0]) ^ ((unsigned long long) p1 * c_ple_mult[1]);
    if (hd >= PLE_NH / 2) mixed ^= (unsigned long long) p2 * c_ple_mult[2];
    rows[t * PLE_NH + hd] = mixed % c_ple_size[hd] + c_ple_off[hd];
}

// gather and dequantize the 16 rows (IQ4_NL, 160 values each) from the pinned host table.  grid nt, 128 threads
__global__ void k_ple_gather(const uint8_t * __restrict__ table, const unsigned long long * __restrict__ rows,
                             float * __restrict__ emb) {
    pdl_wait();
    pdl_trigger();
    const int t = blockIdx.x;
    for (int i = threadIdx.x; i < PLE_NH * (PLE_HD / 32); i += blockDim.x) {
        const int hd = i / (PLE_HD / 32), g = i % (PLE_HD / 32);
        const uint8_t * row = table + rows[t * PLE_NH + hd] * (PLE_HD / 32 * 18);
        float v[32];
        dq32<T_IQ4_NL>(row, g, v);
#pragma unroll
        for (int j = 0; j < 32; ++j) emb[(size_t) t * D + hd * PLE_HD + g * 32 + j] = v[j];
    }
}

// per stream s: gate = <rmsnorm(key_s) wk, rmsnorm(h_s) wq> / sqrt(D), signed sqrt; gv = sigmoid(gate) value;
// gvn = rmsnorm(gv) wc.   grid (nt, HC), 256 threads
__global__ void k_ple_gate(const float * __restrict__ key, const float * __restrict__ val, const float * __restrict__ h,
                           const float * __restrict__ wk, const float * __restrict__ wq, const float * __restrict__ wc,
                           float * __restrict__ gv, float * __restrict__ gvn) {
    pdl_wait();
    pdl_trigger();
    __shared__ float red[8];
    const int t = blockIdx.x, s = blockIdx.y;
    const float * kt = key + (size_t) t * HCD + s * D;
    const float * ht = h + (size_t) t * HCD + s * D;
    const float * vt = val + (size_t) t * D;
    float sk = 0.0f, sq = 0.0f;
    for (int c = threadIdx.x; c < D; c += 256) { sk += kt[c] * kt[c]; sq += ht[c] * ht[c]; }
    sk = block_sum<8>(sk, red);
    sq = block_sum<8>(sq, red);
    const float rk = rsqrtf(sk / D + EPS), rq = rsqrtf(sq / D + EPS);
    float dot = 0.0f;
    for (int c = threadIdx.x; c < D; c += 256) dot += (kt[c] * rk * wk[s * D + c]) * (ht[c] * rq * wq[s * D + c]);
    dot = block_sum<8>(dot, red);
    float gate = dot / sqrtf((float) D);
    gate = gate == 0.0f ? 0.0f : copysignf(sqrtf(fmaxf(fabsf(gate), 1e-6f)), gate);
    const float sgm = sigm(gate);
    float sv = 0.0f;
    for (int c = threadIdx.x; c < D; c += 256) {
        const float x = sgm * vt[c];
        gv[(size_t) t * HCD + s * D + c] = x;
        sv += x * x;
    }
    sv = block_sum<8>(sv, red);
    const float rv = rsqrtf(sv / D + EPS);
    for (int c = threadIdx.x; c < D; c += 256)
        gvn[(size_t) t * HCD + s * D + c] = gv[(size_t) t * HCD + s * D + c] * rv * wc[s * D + c];
}

// dilated depthwise conv (kernel 4, dilation 3) over the history of gvn, silu, then h += gv + conv; history advanced.
// n = HCD threads
__global__ void k_ple_conv(float * __restrict__ h, const float * __restrict__ gv, const float * __restrict__ gvn,
                           float * __restrict__ hist, const float * __restrict__ w, int nt) {
    pdl_wait();
    pdl_trigger();
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= HCD) return;
    float x[PLE_HIST + MAXT];
#pragma unroll
    for (int i = 0; i < PLE_HIST; ++i) x[i] = hist[(size_t) i * HCD + c];
    for (int t = 0; t < nt; ++t) x[PLE_HIST + t] = gvn[(size_t) t * HCD + c];
    const float w0 = w[c * 4 + 0], w1 = w[c * 4 + 1], w2 = w[c * 4 + 2], w3 = w[c * 4 + 3];
    for (int t = 0; t < nt; ++t) {
        const int i = PLE_HIST + t;
        const float s = w0 * x[i - 9] + w1 * x[i - 6] + w2 * x[i - 3] + w3 * x[i];
        h[(size_t) t * HCD + c] += gv[(size_t) t * HCD + c] + silu(s);
    }
#pragma unroll
    for (int i = 0; i < PLE_HIST; ++i) hist[(size_t) i * HCD + c] = x[nt + i];
}

// tok_all[pos + t] = ids[t]
__global__ void k_store_tokens(const int * __restrict__ ids, int * __restrict__ tok_all, const int * __restrict__ d_pos, int nt) {
    pdl_wait();
    pdl_trigger();
    if ((int) threadIdx.x < nt) tok_all[*d_pos + threadIdx.x] = ids[threadIdx.x];
}

__global__ void k_advance(int * __restrict__ d_pos, int nt) { *d_pos += nt; }

// ---------------------------------------------------------------------------------------------
// LM head argmax (lowest index wins ties)

__global__ void k_argmax(const float * __restrict__ logits, int * __restrict__ out) {
    pdl_wait();
    pdl_trigger();
    __shared__ float bv[32];
    __shared__ int bi[32];
    const int t = blockIdx.x;
    const float * l = logits + (size_t) t * VOCAB;
    float v = -INFINITY;
    int idx = 0;
    for (int i = threadIdx.x; i < VOCAB; i += blockDim.x)
        if (l[i] > v) { v = l[i]; idx = i; }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        const float v2 = __shfl_xor_sync(0xffffffffu, v, o);
        const int i2 = __shfl_xor_sync(0xffffffffu, idx, o);
        if (v2 > v || (v2 == v && i2 < idx)) { v = v2; idx = i2; }
    }
    if ((threadIdx.x & 31) == 0) { bv[threadIdx.x >> 5] = v; bi[threadIdx.x >> 5] = idx; }
    __syncthreads();
    if (threadIdx.x == 0) {
        for (int j = 1; j < (int) (blockDim.x >> 5); ++j)
            if (bv[j] > v || (bv[j] == v && bi[j] < idx)) { v = bv[j]; idx = bi[j]; }
        // thread 0 holds warp 0's result in v/idx already
        out[t] = idx;
    }
}

// ---------------------------------------------------------------------------------------------
// host: weights and engine

struct QW {            // a 2D (or stacked 3D) quantized weight resident on one device, in the tiled layout
    uint32_t type = 0;
    uint64_t rb = 0;   // bytes per GGUF row
    int rows = 0, cols = 0;
    int cv = 0, cb = 0, nch = 0, ntiles = 0;  // values per chunk, bytes per row of a unit, chunks per row, 16-row tiles
    int wpt = 1, ks = 1, tb = 1, nbuf = 0;    // launch plan; from the shape only, the same for every batch size
    uint8_t * d = nullptr;
    float * part = nullptr;                    // split-K partials [ks][MAXT][ntiles * 16] (the card's shared scratch)
};

struct LayerW {
    int dev = 0;
    bool attn = false, ple = false;
    float * hca_norm = nullptr, * hcf_norm = nullptr;
    QW hca_dinj, hca_up, hcf_dinj, hcf_up;       // hyper-connection down (320 rows) + inject (4 rows) stacked
    // DeltaNet
    QW qkv, z, ab, ssm_out;                      // ab = alpha (48 rows) + beta (48 rows) stacked
    float * conv_w = nullptr, * dt = nullptr, * ssm_a = nullptr, * ssm_norm = nullptr;
    // attention + indexer
    QW q, k, v, o, ixqk;                         // ixqk = indexer q (512 rows) + k (128 rows) stacked
    float * qn = nullptr, * kn = nullptr, * ixqn = nullptr, * ixkn = nullptr;
    // MoE
    QW rsg, sh_gate, sh_up, sh_down, eg, eu, ed;  // rsg = router (512 rows) + shared-expert gate (1 row) stacked
    // per-layer embedding
    QW ple_key, ple_val;
    float * ple_nk = nullptr, * ple_nq = nullptr, * ple_nc = nullptr, * ple_conv = nullptr;
    // state
    float * conv_state = nullptr, * S = nullptr;                       // DeltaNet
    float * Kc = nullptr, * Vc = nullptr, * IKc = nullptr, * BKc = nullptr;  // attention
    float * ple_hist = nullptr;
};

struct Dev {           // per-GPU working buffers
    int id = 0;
    cudaStream_t s = nullptr;
    int * d_pos = nullptr;                 // tokens already in this card's caches
    cudaGraphExec_t graph[MAXT + 1] = {};  // one captured forward per block size
    float * h = nullptr, * hn = nullptr, * x = nullptr, * y = nullptr, * t320 = nullptr, * zinj = nullptr, * inj = nullptr;
    float * qkv = nullptr, * zz = nullptr, * on = nullptr;
    float * qg = nullptr, * kk = nullptr, * vv = nullptr, * Q = nullptr, * og = nullptr, * part_o = nullptr, * part_ml = nullptr;
    float * iq = nullptr, * ik = nullptr, * Qix = nullptr, * ixs = nullptr;
    uint8_t * ixmask = nullptr;
    float * rlog = nullptr, * wsel = nullptr, * act = nullptr, * shg = nullptr, * shu = nullptr, * sha = nullptr,
          * shd = nullptr, * dpart = nullptr;
    float * lr = nullptr, * ab = nullptr, * ixqk = nullptr;  // outputs of stacked weights (row stride LRS, 2 NVH, IXQK)
    int * sel = nullptr;
};

struct Engine {
    Gguf g1, g2;
    int split = 24;                       // layers [0, split) on GPU0, the rest on GPU1
    int max_ctx = 0, n_past = 0, maxb = 0;
    Dev dv[2];
    LayerW L[NL];
    uint8_t * tok_embd = nullptr;         // GPU0
    uint64_t tok_rb = 0;
    float * out_norm = nullptr;           // GPU1: final mixer + head
    QW out_down, out_up, head;
    float * logits = nullptr;
    int * d_argmax = nullptr;
    int * tok_all = nullptr;              // GPU0: every token id so far (n-gram context)
    int * d_ids = nullptr;                // GPU0: ids of the current block
    unsigned long long * ple_rows = nullptr;
    float * ple_emb = nullptr, * ple_key = nullptr, * ple_val = nullptr, * ple_gv = nullptr, * ple_gvn = nullptr;
    uint8_t * ngram = nullptr;            // pinned host, mapped for GPU0
    float * hbox = nullptr;               // pinned host mailbox GPU0 -> GPU1 (residual stream)
    int * in_tok = nullptr, * out_am = nullptr;  // pinned host: block tokens in, argmax out
    cudaEvent_t ev_handoff = nullptr;
    uint8_t * stage[2] = {nullptr, nullptr};  // load only: GGUF bytes of one tensor before the tiled repack
    size_t stage_n = 0;
    bool use_graphs = true;
    size_t bytes_dev[2] = {0, 0};
    float * dbg = nullptr;                // optional host buffer: h after every layer [NL][nt][HCD]
    double last_ms = 0;
};

static Engine * g_eng = nullptr;

template <typename T> static int dalloc(Engine & e, int dev, T ** p, size_t n) {
    CK(cudaSetDevice(dev));
    CK(cudaMalloc((void **) p, n * sizeof(T) + 16));
    CK(cudaMemset(*p, 0, n * sizeof(T) + 16));
    e.bytes_dev[dev] += n * sizeof(T);
    return 0;
}

static const GTensor * find(Engine & e, const std::string & name) {
    auto it = e.g1.tensors.find(name);
    if (it != e.g1.tensors.end()) return &it->second;
    it = e.g2.tensors.find(name);
    return it == e.g2.tensors.end() ? nullptr : &it->second;
}

// ---------------------------------------------------------------------------------------------
// tiled layout and GEMV launch plan

constexpr int TG_SMEM = 48 * 1024;         // dynamic shared memory per block (no opt-in, two blocks fit an SM)
constexpr int TG_PART = MAXT * 65536;      // floats of split-K scratch per card
static int nsm_g = 170;
static int g_tg_warps = 16;                // target warps per SM for one GEMV
static int g_ov_wpt = 0, g_ov_ks = 0, g_ov_tb = 0;  // benchmark overrides of the plan (0 = automatic)
static int g_tg_waves = 1;                // persistent GEMV grid: resident blocks x this (0 = one block per tile group)

// values per chunk: BF16 64 (128-byte rows for the register path), else the largest of 256/128/64/32 that is a whole
// number of blocks and divides the row
static int pick_cv(uint32_t type, int cols) {
    if (type == T_BF16) return cols % 64 ? 0 : 64;
    int qk, b;
    if (!type_info(type, qk, b) || qk == 1) return 0;
    for (int cv = 256; cv >= 32; cv >>= 1)
        if (cv % qk == 0 && cols % cv == 0) return cv;
    return 0;
}

static size_t tg_smem(const QW & w, int tb, int wpt, int nbuf) {
    return (size_t) tb * wpt * ((size_t) nbuf * 16 * w.cb + 2 * 16 * MAXT * 4) + 4 * grid_words(w.type);
}

// resident blocks per SM of a kernel at a launch shape (cached)
static int resident_blocks(const void * fn, int threads, size_t smem) {
    static std::map<std::tuple<const void *, int, size_t>, int> cache;
    const auto key = std::make_tuple(fn, threads, smem);
    auto it = cache.find(key);
    if (it != cache.end()) return it->second;
    int nb = 1;
    if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, fn, threads, smem) != cudaSuccess) nb = 1;
    return cache[key] = std::max(1, nb);
}

// Grid of tile groups (tb tiles; ks blocks each). Up to 3 GPU-fulls of groups: persistent, as many blocks as fit, so
// no partial last wave runs alone. Beyond that: one block per group, so blocks that finish early take the next group
// (a static split of many groups leaves the slowest block's extra group as the tail).
static dim3 tg_grid(const QW & w, const void * fn, size_t smem) {
    const int ngroups = (w.ntiles + w.tb - 1) / w.tb;
    const int fit = std::max(1, resident_blocks(fn, w.tb * w.wpt * 32, smem) * nsm_g / w.ks);
    return dim3(g_tg_waves && ngroups <= 3 * fit ? std::min(ngroups, fit) : ngroups, w.ks);
}

// Layout and launch plan of a rows x cols weight. S splits per tile (wpt warps in a block x ks blocks) bring the GEMV
// to about g_tg_warps warps per SM; a tile's splits stay in one block (no partials) while its warps fit the
// shared-memory budget. Depends on the shape and format only, never on the number of tokens.
static bool plan_qw(QW & w, uint32_t type, int rows, int cols) {
    w.type = type;
    w.rows = rows;
    w.cols = cols;
    w.rb = row_bytes(type, cols);
    w.cv = pick_cv(type, cols);
    if (!w.rb || !w.cv) return false;
    w.cb = (int) row_bytes(type, w.cv);
    w.nch = cols / w.cv;
    w.ntiles = (rows + 15) / 16;
    const bool bf = type == T_BF16;
    const int S = std::max(1, std::min(w.nch, (nsm_g * g_tg_warps + w.ntiles - 1) / w.ntiles));
    auto fit = [&](int nbuf) {
        int m = 8;
        while (m > 1 && tg_smem(w, 1, m, bf ? 0 : nbuf) > TG_SMEM) --m;
        return m;
    };
    for (int nbuf = 1; nbuf <= 2; ++nbuf) {
        const int wmax = fit(nbuf);
        w.ks = (S + wmax - 1) / wmax;
        w.wpt = S / w.ks;
        if (bf || (w.nch + S - 1) / S < 2 || nbuf == 2) break;
    }
    if (g_ov_wpt) w.wpt = std::min(std::min(g_ov_wpt, 8), w.nch);
    if (g_ov_ks) w.ks = std::max(1, std::min(g_ov_ks, w.nch / w.wpt));
    const int cpw = (w.nch + w.wpt * w.ks - 1) / (w.wpt * w.ks);
    w.nbuf = bf ? 0 : (cpw > 1 ? 2 : 1);
    if (w.nbuf == 2 && tg_smem(w, 1, w.wpt, 2) > TG_SMEM) w.nbuf = 1;
    w.tb = g_ov_tb ? g_ov_tb : std::max(1, 8 / w.wpt);
    w.tb = std::min(w.tb, std::max(1, 8 / w.wpt));
    while (w.tb > 1 && tg_smem(w, w.tb, w.wpt, w.nbuf) > TG_SMEM) --w.tb;
    if (tg_smem(w, w.tb, w.wpt, w.nbuf) > TG_SMEM) return false;
    return w.ks == 1 || (size_t) w.ks * MAXT * w.ntiles * 16 <= TG_PART;
}

static size_t tiled_bytes(const QW & w) { return (size_t) w.ntiles * w.nch * 16 * w.cb; }

// GGUF rows on the device (src) -> the tiled layout in w.d (current device, default stream)
static void repack(const QW & w, const uint8_t * src, bool hcperm = false) {
    const uint64_t nunits = (uint64_t) w.ntiles * w.nch;
    k_repack<<<(unsigned) ((nunits + 7) / 8), 256>>>(src, w.d, w.rows, w.rb, w.cb, w.nch, nunits, hcperm ? 1 : 0);
}

// load a weight with rows = ne[1] * ne[2] rows of ne[0] columns, in its GGUF storage type, tiled on the device;
// hcperm: a hyper-connection up weight for gemv_mix (rows in hc_perm_row order)
static int load_qw(Engine & e, int dev, const std::string & name, QW & w, int cols, int rows, bool hcperm = false) {
    const GTensor * t = find(e, name);
    if (!t) FAIL("missing tensor " + name);
    if ((int) t->ne[0] != cols || (int) (t->ne[1] * t->ne[2]) != rows) FAIL(name + ": unexpected shape");
    if (!plan_qw(w, t->type, rows, cols)) FAIL(name + ": unsupported type/shape " + std::to_string(t->type));
    if (t->nbytes != w.rb * rows || t->nbytes > e.stage_n) FAIL(name + ": unexpected byte size");
    if (dalloc(e, dev, &w.d, tiled_bytes(w))) return -1;
    if (hcperm && (w.type != T_BF16 || w.ks != 1 || rows != HCD)) FAIL(name + ": hyper-connection up weight must be BF16, ks 1");
    CK(cudaMemcpy(e.stage[dev], t->host, t->nbytes, cudaMemcpyHostToDevice));
    repack(w, e.stage[dev], hcperm);
    CK(cudaGetLastError());
    if (w.ks > 1 && dalloc(e, dev, &w.part, (size_t) w.ks * MAXT * w.ntiles * 16)) return -1;
    return 0;
}

// GGUF tensors of one format and row length stacked into one tiled weight, rows[i] rows each (all but the last a
// multiple of 16, so every tile holds rows of one tensor): one GEMV for weights that read the same input
static int load_qw_cat(Engine & e, int dev, const std::vector<std::string> & names, const std::vector<int> & rows, QW & w,
                       int cols) {
    int total = 0;
    const GTensor * t0 = find(e, names[0]);
    if (!t0) FAIL("missing tensor " + names[0]);
    for (size_t i = 0; i < names.size(); ++i) {
        if (i + 1 < names.size() && rows[i] % 16) FAIL(names[i] + ": stacked rows must be a multiple of 16");
        total += rows[i];
    }
    if (!plan_qw(w, t0->type, total, cols)) FAIL(names[0] + ": unsupported type/shape " + std::to_string(t0->type));
    if (dalloc(e, dev, &w.d, tiled_bytes(w))) return -1;
    for (size_t i = 0, r0 = 0; i < names.size(); r0 += rows[i], ++i) {
        const GTensor * t = find(e, names[i]);
        if (!t) FAIL("missing tensor " + names[i]);
        if (t->type != w.type || (int) t->ne[0] != cols || (int) (t->ne[1] * t->ne[2]) != rows[i] ||
            t->nbytes != w.rb * rows[i] || t->nbytes > e.stage_n)
            FAIL(names[i] + ": does not stack with " + names[0]);
        QW piece = w;
        piece.rows = rows[i];
        piece.ntiles = (rows[i] + 15) / 16;
        piece.d = w.d + r0 / 16 * w.nch * 16 * w.cb;
        CK(cudaMemcpy(e.stage[dev], t->host, t->nbytes, cudaMemcpyHostToDevice));
        repack(piece, e.stage[dev]);
        CK(cudaGetLastError());
    }
    if (w.ks > 1 && dalloc(e, dev, &w.part, (size_t) w.ks * MAXT * w.ntiles * 16)) return -1;
    return 0;
}

// f32 (or f16) vector/matrix converted to f32 on device; `sub` is subtracted from every value
static int load_f32(Engine & e, int dev, const std::string & name, size_t n, float ** out) {
    const GTensor * t = find(e, name);
    if (!t) FAIL("missing tensor " + name);
    if (t->ne[0] * t->ne[1] * t->ne[2] != n) FAIL(name + ": unexpected size");
    std::vector<float> h(n);
    if (t->type == T_F32) memcpy(h.data(), t->host, n * 4);
    else if (t->type == T_F16) for (size_t i = 0; i < n; ++i) h[i] = __half2float(__ushort_as_half(((const uint16_t *) t->host)[i]));
    else FAIL(name + ": expected f32/f16");
    if (dalloc(e, dev, out, n)) return -1;
    CK(cudaMemcpy(*out, h.data(), n * 4, cudaMemcpyHostToDevice));
    return 0;
}

static int alloc_dev_buffers(Engine & e, Dev & d) {
    const int nsplit = NSPLIT;
#define A(p, n) if (dalloc(e, d.id, &d.p, (size_t) (n))) return -1
    A(d_pos, 1); A(h, MAXT * HCD); A(hn, MAXT * HCD); A(x, MAXT * D); A(y, MAXT * D); A(t320, MAXT * HCR);
    A(zinj, MAXT * HC); A(inj, MAXT * HC);
    A(qkv, MAXT * CONVCH); A(zz, MAXT * DINNER); A(on, MAXT * DINNER);
    A(qg, MAXT * NH * QG); A(kk, MAXT * NKV * HD); A(vv, MAXT * NKV * HD); A(Q, MAXT * NH * HD); A(og, MAXT * NH * HD);
    A(part_o, (size_t) MAXT * NH * nsplit * HD); A(part_ml, (size_t) MAXT * NH * nsplit * 2);
    A(iq, MAXT * IXH * IXD); A(ik, MAXT * IXD); A(Qix, MAXT * IXH * IXD); A(ixs, (size_t) MAXT * e.maxb);
    A(ixmask, (size_t) MAXT * e.maxb);
    A(rlog, MAXT * RSG); A(sel, MAXT * NEU); A(wsel, MAXT * NEU); A(act, MAXT * NEU * FFE);
    A(shg, MAXT * FFE); A(shu, MAXT * FFE); A(sha, MAXT * FFE); A(shd, MAXT * D);
    A(dpart, (size_t) MAXT * NEU * D); A(lr, MAXT * LRS); A(ab, MAXT * 2 * NVH); A(ixqk, MAXT * IXQK);
#undef A
    return 0;
}

// ---------------------------------------------------------------------------------------------
// launchers

static bool g_pdl = true;    // launch kernels with programmatic stream serialization
static bool g_early = true;  // let GEMVs marked early run beside the kernel before them

template <typename... P, typename... A>
static void launch_pdl(void (*k)(P...), dim3 grid, int threads, size_t smem, cudaStream_t s, A... args) {
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = grid;
    cfg.blockDim = dim3(threads);
    cfg.dynamicSmemBytes = smem;
    cfg.stream = s;
    cudaLaunchAttribute at[1];
    at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    at[0].val.programmaticStreamSerializationAllowed = 1;
    cfg.attrs = at;
    cfg.numAttrs = g_pdl ? 1 : 0;
    cudaLaunchKernelEx(&cfg, k, ((P) args)...);
}

template <uint32_t TY, int CV>
static void tgemv_t(const QW & w, const float * x, int xs, float * y, int ys, int nt, bool acc, cudaStream_t s, bool early) {
    if constexpr (CV % QT<TY>::qk == 0) {
        const size_t smem = tg_smem(w, w.tb, w.wpt, w.nbuf);
        auto k = nt == 1 ? k_tgemv<TY, CV, 1> : k_tgemv<TY, CV, MAXT>;
        launch_pdl(k, tg_grid(w, (const void *) k, smem), w.tb * w.wpt * 32, smem, s, w.d, w.ntiles, w.nch, w.rows, w.wpt,
                   w.ks, w.nbuf, x, xs, y, ys, nt, acc ? 1 : 0, w.part, early ? 1 : 0);
    }
}

template <uint32_t TY>
static void tgemv_cv(const QW & w, const float * x, int xs, float * y, int ys, int nt, bool acc, cudaStream_t s, bool early) {
    switch (w.cv) {
        case 256: tgemv_t<TY, 256>(w, x, xs, y, ys, nt, acc, s, early); break;
        case 128: tgemv_t<TY, 128>(w, x, xs, y, ys, nt, acc, s, early); break;
        case 64:  tgemv_t<TY, 64>(w, x, xs, y, ys, nt, acc, s, early); break;
        case 32:  tgemv_t<TY, 32>(w, x, xs, y, ys, nt, acc, s, early); break;
        default: break;
    }
}

// y[t][r] (+)= W[r] . x[t] for t < nt (x and y token strides xs, ys). Returns the result as the consumer reads it:
// with fold, split-K partials are left for the consumer to add (no reduce kernel), otherwise y.
// early: x was complete before the previous kernel started (it was written two or more kernels back, and the previous
// kernel is not itself early), so this GEMV reads it without waiting and runs beside the previous kernel; it waits for
// the previous kernel only before it exits, so its completion still implies the previous kernel's.
static GOut gemv(const QW & w, const float * x, int xs, float * y, int ys, int nt, cudaStream_t s, bool acc = false,
                 bool fold = false, bool early = false) {
    early = early && g_early && g_pdl;
    switch (w.type) {
        case T_BF16: {
            const size_t smem = tg_smem(w, w.tb, w.wpt, 0);
            auto k = nt == 1 ? k_tgemv_bf16<1, false> : k_tgemv_bf16<MAXT, false>;
            launch_pdl(k, tg_grid(w, (const void *) k, smem), w.tb * w.wpt * 32, smem, s, w.d, w.ntiles, w.nch, w.rows,
                       w.wpt, w.ks, x, xs, y, ys, nt, acc ? 1 : 0, w.part, early ? 1 : 0, (const float *) nullptr,
                       (const float *) nullptr, (float *) nullptr, (float *) nullptr);
            break;
        }
        case T_Q8_0:    tgemv_cv<T_Q8_0>(w, x, xs, y, ys, nt, acc, s, early); break;
        case T_Q4_K:    tgemv_cv<T_Q4_K>(w, x, xs, y, ys, nt, acc, s, early); break;
        case T_Q5_K:    tgemv_cv<T_Q5_K>(w, x, xs, y, ys, nt, acc, s, early); break;
        case T_Q6_K:    tgemv_cv<T_Q6_K>(w, x, xs, y, ys, nt, acc, s, early); break;
        case T_IQ2_XXS: tgemv_cv<T_IQ2_XXS>(w, x, xs, y, ys, nt, acc, s, early); break;
        case T_IQ2_XS:  tgemv_cv<T_IQ2_XS>(w, x, xs, y, ys, nt, acc, s, early); break;
        case T_IQ2_S:   tgemv_cv<T_IQ2_S>(w, x, xs, y, ys, nt, acc, s, early); break;
        case T_IQ3_XXS: tgemv_cv<T_IQ3_XXS>(w, x, xs, y, ys, nt, acc, s, early); break;
        case T_IQ3_S:   tgemv_cv<T_IQ3_S>(w, x, xs, y, ys, nt, acc, s, early); break;
        case T_IQ4_NL:  tgemv_cv<T_IQ4_NL>(w, x, xs, y, ys, nt, acc, s, early); break;
        case T_IQ4_XS:  tgemv_cv<T_IQ4_XS>(w, x, xs, y, ys, nt, acc, s, early); break;
        case T_Q2_0:    tgemv_cv<T_Q2_0>(w, x, xs, y, ys, nt, acc, s, early); break;
        default: break;
    }
    if (w.ks > 1) {
        if (fold && !acc) return GOut{w.part, w.ks, w.ntiles * 16, (size_t) MAXT * w.ntiles * 16};
        launch_pdl(k_tg_reduce, dim3((w.rows + 255) / 256, nt), 256, 0, s, w.part, w.ks, w.ntiles * 16, w.rows, y, ys, acc ? 1 : 0);
    }
    return GOut{y, 1, ys, 0};
}

// hyper-connection up GEMV (BF16, rows in hc_perm_row order, ks 1) with the mix as its epilogue: x and (zinj
// non-null) inj, from u = up . t320 and hn
static void gemv_mix(const QW & up, const float * t320, const float * hn, const float * zinj, float * xo, float * injo,
                     int nt, cudaStream_t s) {
    const size_t smem = tg_smem(up, up.tb, up.wpt, 0);
    auto k = nt == 1 ? k_tgemv_bf16<1, true> : k_tgemv_bf16<MAXT, true>;
    launch_pdl(k, tg_grid(up, (const void *) k, smem), up.tb * up.wpt * 32, smem, s, up.d, up.ntiles, up.nch, up.rows,
               up.wpt, up.ks, t320, HCR, (float *) nullptr, 0, nt, 0, up.part, 0, hn, zinj, xo, injo);
}

// routed-expert launch shapes, fixed (so independent of the batch): tb tiles x wpt warps per block, for gate+up and down
static int g_gu_wpt = 5, g_gu_tb = 1, g_dn_wpt = 2, g_dn_tb = 4;
static int g_moe_var = 1;  // routed-expert gate/up kernel: 1 = 8-row tiles (k_tmoe_gu8), 0 = 16-row tiles (k_tmoe_gu); A/B

template <uint32_t TY>
static void moe_gu_t(const QW & g, const QW & u, const int * sel, const float * x, float * act, int npairs, cudaStream_t s) {
    constexpr int UB = 16 * qcb<TY, 256>(), NCH = D / 256;
    int wpt = std::min(g_gu_wpt, 8);
    auto smem = [&](int tb) {
        return (size_t) tb * wpt * (((NCH + wpt - 1) / wpt > 1 ? 2 : 1) * 2 * UB + 16 * 2 * 4) + 4 * GRID<TY>::words;
    };
    while (wpt > 1 && smem(1) > TG_SMEM) --wpt;
    int tb = std::min(g_gu_tb, 8 / wpt);
    while (tb > 1 && smem(tb) > TG_SMEM) --tb;
    if (g_moe_var & 1) {  // 8-row half tiles
        constexpr int HB = 8 * qcb<TY, 256>();
        auto smem8 = [&](int tb8) {
            return (size_t) tb8 * wpt * (((NCH + wpt - 1) / wpt > 1 ? 2 : 1) * 2 * HB + 16 * 4) + 4 * GRID<TY>::words;
        };
        int tb8 = std::min(g_gu_tb, 8 / wpt);
        while (tb8 > 1 && smem8(tb8) > TG_SMEM) --tb8;
        launch_pdl(k_tmoe_gu8<TY>, dim3((FFE / 8 + tb8 - 1) / tb8, npairs), tb8 * wpt * 32, smem8(tb8), s, g.d, u.d, sel, x, act, wpt);
        return;
    }
    launch_pdl(k_tmoe_gu<TY>, dim3((FFE / 16 + tb - 1) / tb, npairs), tb * wpt * 32, smem(tb), s, g.d, u.d, sel, x, act, wpt);
}

static void moe_gu(const QW & g, const QW & u, const int * sel, const float * x, float * act, int npairs, cudaStream_t s) {
    switch (g.type) {
        case T_IQ2_XXS: moe_gu_t<T_IQ2_XXS>(g, u, sel, x, act, npairs, s); break;
        case T_IQ2_XS:  moe_gu_t<T_IQ2_XS>(g, u, sel, x, act, npairs, s); break;
        case T_IQ2_S:   moe_gu_t<T_IQ2_S>(g, u, sel, x, act, npairs, s); break;
        case T_IQ3_XXS: moe_gu_t<T_IQ3_XXS>(g, u, sel, x, act, npairs, s); break;
        case T_IQ3_S:   moe_gu_t<T_IQ3_S>(g, u, sel, x, act, npairs, s); break;
        default: break;
    }
}

template <uint32_t TY>
static void moe_down_t(const QW & w, const int * sel, const float * act, float * dpart, int npairs, cudaStream_t s) {
    constexpr int UB = 16 * qcb<TY, 128>(), NCH = FFE / 128;
    int wpt = std::min(g_dn_wpt, 8);
    auto smem = [&](int tb) { return (size_t) tb * wpt * (((NCH + wpt - 1) / wpt > 1 ? 2 : 1) * UB + 16 * 4); };
    while (wpt > 1 && smem(1) > TG_SMEM) --wpt;
    int tb = std::min(g_dn_tb, 8 / wpt);
    while (tb > 1 && smem(tb) > TG_SMEM) --tb;
    launch_pdl(k_tmoe_down<TY>, dim3((D / 16 + tb - 1) / tb, npairs), tb * wpt * 32, smem(tb), s, w.d, sel, act, dpart, wpt);
}

static void moe_down(const QW & w, const int * sel, const float * act, float * dpart, int npairs, cudaStream_t s) {
    switch (w.type) {
        case T_Q2_0:   moe_down_t<T_Q2_0>(w, sel, act, dpart, npairs, s); break;
        case T_IQ4_NL: moe_down_t<T_IQ4_NL>(w, sel, act, dpart, npairs, s); break;
        default: break;
    }
}

// hyper-connection read: hn = norm(h) (unless the residual write before already left it), x = mix, inj = injection
// weights. dinj = down rows (+ the 4 inject rows, unless this is the final mixer)
static void hc_read(Dev & d, const float * norm, const QW & dinj, const QW & up, int nt, bool normed) {
    cudaStream_t s = d.s;
    const bool inj = dinj.rows > HCR;
    if (!normed) launch_pdl(k_groupnorm, dim3(nt, HC), D / 4, 0, s, d.h, norm, d.hn, HCD);
    const GOut lr = gemv(dinj, d.hn, HCD, d.lr, LRS, nt, s, false, true);
    launch_pdl(k_hc_lowrank, nt, 384, 0, s, lr, d.t320, d.zinj, inj ? 1 : 0);
    gemv_mix(up, d.t320, d.hn, inj ? d.zinj : nullptr, d.x, d.inj, nt, s);
}

// the RMSNorm weight of the hyper-connection read after layer il on the same card (the next layer's attention read,
// or the final mixer's), or null when that read normalizes for itself (the n-gram layer changes h first; the other
// card starts from the residual stream it receives)
static const float * next_norm(const Engine & e, int il) {
    if (il + 1 == NL) return e.out_norm;
    if (il + 1 == e.split || e.L[il + 1].ple) return nullptr;
    return e.L[il + 1].hca_norm;
}

static void layer_forward(Engine & e, int il, int nt) {
    LayerW & L = e.L[il];
    Dev & d = e.dv[L.dev];
    cudaStream_t s = d.s;
    if (L.ple) {
        launch_pdl(k_ple_ids, nt, 32, 0, s, e.tok_all, d.d_pos, nt, e.ple_rows);
        launch_pdl(k_ple_gather, nt, 128, 0, s, e.ngram, e.ple_rows, e.ple_emb);
        gemv(L.ple_key, e.ple_emb, D, e.ple_key, HCD, nt, s);
        gemv(L.ple_val, e.ple_emb, D, e.ple_val, D, nt, s);
        launch_pdl(k_ple_gate, dim3(nt, HC), 256, 0, s, e.ple_key, e.ple_val, d.h, L.ple_nk, L.ple_nq, L.ple_nc, e.ple_gv, e.ple_gvn);
        launch_pdl(k_ple_conv, HCD / 256, 256, 0, s, d.h, e.ple_gv, e.ple_gvn, L.ple_hist, L.ple_conv, nt);
    }
    // layer 0, the n-gram layer and the first layer of a card normalize for themselves
    hc_read(d, L.hca_norm, L.hca_dinj, L.hca_up, nt, !(il == 0 || L.ple || il == e.split));
    GOut y;
    if (!L.attn) {
        gemv(L.qkv, d.x, D, d.qkv, CONVCH, nt, s);  // z and alpha/beta read the same x: beside it
        gemv(L.z, d.x, D, d.zz, DINNER, nt, s, false, false, true);
        const GOut ab = gemv(L.ab, d.x, D, d.ab, 2 * NVH, nt, s, false, true, true);
        launch_pdl(k_gdn, NVH, DS, 0, s, d.qkv, L.conv_state, L.conv_w, ab, L.dt, L.ssm_a, L.S, d.zz, L.ssm_norm, d.on, nt);
        y = gemv(L.ssm_out, d.on, DINNER, d.y, D, nt, s, false, true);
    } else {
        gemv(L.q, d.x, D, d.qg, NH * QG, nt, s);  // k, v and the indexer read the same x: beside it
        const GOut kk = gemv(L.k, d.x, D, d.kk, NKV * HD, nt, s, false, true, true);
        const GOut vv = gemv(L.v, d.x, D, d.vv, NKV * HD, nt, s, false, true, true);
        const GOut iqk = gemv(L.ixqk, d.x, D, d.ixqk, IXQK, nt, s, false, true, true);
        launch_pdl(k_qk_norm_rope, dim3(nt, NH + 2 * NKV), HD, 0, s, d.qg, kk, vv, L.qn, L.kn, d.Q, L.Kc, L.Vc, d.d_pos);
        launch_pdl(k_ix_qk, dim3(nt, IXH + 1), IXD, 0, s, iqk, L.ixqn, d.Qix, L.IKc, d.d_pos);
        launch_pdl(k_ix_blocks, nt / IXR + 1, IXD, 0, s, L.IKc, L.ixkn, L.BKc, d.d_pos, nt);
        launch_pdl(k_ix_score, dim3(nt, 64), 256, 0, s, d.Qix, L.BKc, d.ixs, e.maxb, d.d_pos);
        launch_pdl(k_ix_select, nt, 1024, 0, s, d.ixs, e.maxb, d.ixmask, d.d_pos);
        launch_pdl(k_attn_split, dim3(nt, NH, NSPLIT), 256, 0, s, d.Q, L.Kc, L.Vc, d.ixmask, e.maxb, d.part_o, d.part_ml, d.d_pos);
        launch_pdl(k_attn_combine, dim3(nt, NH), HD, 0, s, d.part_o, d.part_ml, d.qg, d.og);
        y = gemv(L.o, d.og, NH * HD, d.y, D, nt, s, false, true);
    }
    launch_pdl(k_hc_add, dim3(nt, HC), D / 4, 0, s, d.h, y, d.inj, (const float *) nullptr, (const float *) nullptr,
               GOut{}, L.hcf_norm, d.hn);

    hc_read(d, L.hcf_norm, L.hcf_dinj, L.hcf_up, nt, true);
    // router, then the shared expert's gate and up beside it (they read the same x); routing + the shared swiglu;
    // routed gate/up, the shared down beside it, routed down
    const GOut rsg = gemv(L.rsg, d.x, D, d.rlog, RSG, nt, s, false, true);
    const GOut sg = gemv(L.sh_gate, d.x, D, d.shg, FFE, nt, s, false, true, true);
    const GOut su = gemv(L.sh_up, d.x, D, d.shu, FFE, nt, s, false, true, true);
    launch_pdl(k_route, nt, NE, 0, s, rsg, d.sel, d.wsel, sg, su, d.sha);
    moe_gu(L.eg, L.eu, d.sel, d.x, d.act, nt * NEU, s);
    const GOut sd = gemv(L.sh_down, d.sha, FFE, d.shd, D, nt, s, false, true, true);
    moe_down(L.ed, d.sel, d.act, d.dpart, nt * NEU, s);
    // MoE output (slot-weighted routed experts + gated shared expert) into the residual, normalized for the next read
    launch_pdl(k_hc_add, dim3(nt, HC), D / 4, 0, s, d.h, sd, d.inj, (const float *) d.dpart, (const float *) d.wsel, rsg,
               next_norm(e, il), d.hn);
}

// GPU0 part of a forward: tokens in, embedding, layers [0, split), residual stream out to the pinned mailbox
static void enqueue_dev0(Engine & e, int nt) {
    Dev & d0 = e.dv[0];
    cudaMemcpyAsync(e.d_ids, e.in_tok, nt * sizeof(int), cudaMemcpyHostToDevice, d0.s);
    launch_pdl(k_store_tokens, 1, MAXT, 0, d0.s, e.d_ids, e.tok_all, d0.d_pos, nt);
    launch_pdl(k_embed, nt, 128, 0, d0.s, e.tok_embd, e.tok_rb, e.d_ids, d0.h);
    for (int il = 0; il < e.split; ++il) {
        layer_forward(e, il, nt);
        if (e.dbg) cudaMemcpyAsync(e.dbg + (size_t) il * nt * HCD, d0.h, (size_t) nt * HCD * 4, cudaMemcpyDeviceToHost, d0.s);
    }
    cudaMemcpyAsync(e.hbox, d0.h, (size_t) nt * HCD * 4, cudaMemcpyDeviceToHost, d0.s);
    k_advance<<<1, 1, 0, d0.s>>>(d0.d_pos, nt);
}

// GPU1 part: residual stream in from the mailbox, layers [split, NL), final mixer, LM head, argmax out
static void enqueue_dev1(Engine & e, int nt) {
    Dev & d1 = e.dv[1];
    cudaMemcpyAsync(d1.h, e.hbox, (size_t) nt * HCD * 4, cudaMemcpyHostToDevice, d1.s);
    for (int il = e.split; il < NL; ++il) {
        layer_forward(e, il, nt);
        if (e.dbg) cudaMemcpyAsync(e.dbg + (size_t) il * nt * HCD, d1.h, (size_t) nt * HCD * 4, cudaMemcpyDeviceToHost, d1.s);
    }
    hc_read(d1, e.out_norm, e.out_down, e.out_up, nt, true);
    gemv(e.head, d1.x, D, e.logits, VOCAB, nt, d1.s);
    launch_pdl(k_argmax, nt, 1024, 0, d1.s, e.logits, e.d_argmax);
    cudaMemcpyAsync(e.out_am, e.d_argmax, nt * sizeof(int), cudaMemcpyDeviceToHost, d1.s);
    k_advance<<<1, 1, 0, d1.s>>>(d1.d_pos, nt);
}

static int capture(Engine & e, int dv, int nt) {
    Dev & d = e.dv[dv];
    CK(cudaSetDevice(dv));
    cudaGraph_t g;
    CK(cudaStreamBeginCapture(d.s, cudaStreamCaptureModeThreadLocal));
    if (dv == 0) enqueue_dev0(e, nt); else enqueue_dev1(e, nt);
    CK(cudaStreamEndCapture(d.s, &g));
    CK(cudaGraphInstantiate(&d.graph[nt], g, 0));
    CK(cudaGraphDestroy(g));
    return 0;
}

// one forward of nt tokens; logits (nt x VOCAB) and/or argmax (nt) copied out when non-null.
// The cards hand off through the event: GPU1's stream waits on GPU0's, the CPU only enqueues.
static int forward(Engine & e, const int * tokens, int nt, float * logits, int * argmax) {
    if (nt < 1 || nt > MAXT) FAIL("nt out of range");
    if (e.n_past + nt > e.max_ctx) FAIL("context full");
    auto t0 = std::chrono::high_resolution_clock::now();
    Dev & d0 = e.dv[0];
    Dev & d1 = e.dv[1];
    memcpy(e.in_tok, tokens, nt * sizeof(int));
    const bool graphs = e.use_graphs && !e.dbg;
    if (graphs && !d0.graph[nt]) {
        if (capture(e, 0, nt) || capture(e, 1, nt)) return -1;
    }
    CK(cudaSetDevice(0));
    if (graphs) CK(cudaGraphLaunch(d0.graph[nt], d0.s)); else enqueue_dev0(e, nt);
    CK(cudaEventRecord(e.ev_handoff, d0.s));
    CK(cudaSetDevice(1));
    CK(cudaStreamWaitEvent(d1.s, e.ev_handoff, 0));
    if (graphs) CK(cudaGraphLaunch(d1.graph[nt], d1.s)); else enqueue_dev1(e, nt);
    if (logits) CK(cudaMemcpyAsync(logits, e.logits, (size_t) nt * VOCAB * 4, cudaMemcpyDeviceToHost, d1.s));
    CK(cudaStreamSynchronize(d1.s));
    CK(cudaGetLastError());
    if (argmax) memcpy(argmax, e.out_am, nt * sizeof(int));
    e.n_past += nt;
    e.last_ms = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - t0).count();
    return 0;
}

// ---------------------------------------------------------------------------------------------
// C API

FN_API const char * fn_error() { return g_err.c_str(); }

FN_API void fn_free() {
    if (!g_eng) return;
    for (int dv = 0; dv < 2; ++dv) {
        cudaSetDevice(dv);
        cudaDeviceSynchronize();
        cudaDeviceReset();
    }
    if (g_eng->ngram) cudaFreeHost(g_eng->ngram);
    delete g_eng;
    g_eng = nullptr;
}

static int read_into(HANDLE hf, uint64_t offset, uint8_t * dst, uint64_t n) {
    const uint64_t CH = 64ull << 20;
    for (uint64_t done = 0; done < n;) {
        const DWORD want = (DWORD) std::min<uint64_t>(CH, n - done);
        OVERLAPPED ov{};
        ov.Offset = (DWORD) ((offset + done) & 0xffffffffu);
        ov.OffsetHigh = (DWORD) ((offset + done) >> 32);
        DWORD got = 0;
        if (!ReadFile(hf, dst + done, want, &got, &ov) || got != want) FAIL("ReadFile failed on the n-gram shard");
        done += got;
    }
    return 0;
}

// split <= 0: choose by free VRAM
FN_API int fn_load(const char * shard1, const char * shard2, int max_ctx, int split) {
    fn_free();
    g_eng = new Engine();
    Engine & e = *g_eng;
    int ndev = 0;
    CK(cudaGetDeviceCount(&ndev));
    if (ndev < 2) FAIL("FAIE2-FN needs two GPUs");
    if (gguf_open(e.g1, shard1) || gguf_open(e.g2, shard2)) return -1;
    if ((int) e.g1.nums["qwen4exp.block_count"] != NL || (int) e.g1.nums["qwen4exp.embedding_length"] != D ||
        (int) e.g1.nums["qwen4exp.expert_count"] != NE || (int) e.g1.nums["qwen4exp.expert_used_count"] != NEU ||
        (int) e.g1.nums["qwen4exp.hyper_connection.count"] != HC)
        FAIL("model hyperparameters differ from this build");
    {
        cudaDeviceProp p;
        CK(cudaGetDeviceProperties(&p, 0));
        nsm_g = p.multiProcessorCount;
    }
    e.max_ctx = (max_ctx + ATT_CH - 1) / ATT_CH * ATT_CH;
    e.maxb = e.max_ctx / IXR;
    // layer bytes -> split
    if (split <= 0) {
        size_t fr[2], tot;
        for (int dv = 0; dv < 2; ++dv) { CK(cudaSetDevice(dv)); CK(cudaMemGetInfo(&fr[dv], &tot)); }
        std::vector<uint64_t> lb(NL, 0);
        for (auto & [name, t] : e.g1.tensors)
            if (name.rfind("blk.", 0) == 0) lb[std::stoi(name.substr(4))] += t.nbytes;
        // fill each card in proportion to its free VRAM (GPU0 also drives the desktop), never past free - reserve
        const uint64_t kv_layer = (uint64_t) e.max_ctx * (2 * NKV * HD + IXD) * 4;
        const uint64_t reserve = 2ull << 30;
        uint64_t total = 0;
        for (int il = 0; il < NL; ++il) total += lb[il] + (is_attn_layer(il) ? kv_layer : 0);
        const uint64_t target = std::min<uint64_t>(fr[0] - reserve, (uint64_t) ((double) total * fr[0] / (fr[0] + fr[1])));
        uint64_t used0 = e.g1.tensors["token_embd.weight"].nbytes;
        split = 0;
        while (split < NL) {
            const uint64_t add = lb[split] + (is_attn_layer(split) ? kv_layer : 0);
            if (used0 + add > target) break;
            used0 += add;
            ++split;
        }
    }
    e.split = split;
    if (split < PLE_LAYER + 1 || split >= NL) FAIL("bad layer split " + std::to_string(split));
    for (int dv = 0; dv < 2; ++dv) {
        e.dv[dv].id = dv;
        CK(cudaSetDevice(dv));
        CK(cudaStreamCreateWithFlags(&e.dv[dv].s, cudaStreamNonBlocking));
        if (alloc_dev_buffers(e, e.dv[dv])) return -1;
    }
    // rope frequencies as transformers computes them (float32)
    float inv[NROT / 2];
    for (int i = 0; i < NROT / 2; ++i) inv[i] = (float) (1.0 / pow((double) ROPE_BASE, (double) (2 * i) / NROT));
    for (int dv = 0; dv < 2; ++dv) { CK(cudaSetDevice(dv)); CK(cudaMemcpyToSymbol(c_inv_freq, inv, sizeof(inv))); }

    // staging for the tiled repack: one tensor at a time, sized to the largest
    for (auto & [name, t] : e.g1.tensors) e.stage_n = std::max<size_t>(e.stage_n, t.nbytes);
    for (int dv = 0; dv < 2; ++dv) { CK(cudaSetDevice(dv)); CK(cudaMalloc(&e.stage[dv], e.stage_n)); }

    // global weights
    {
        const GTensor * t = find(e, "token_embd.weight");
        if (!t || t->type != T_IQ3_S) FAIL("token_embd: expected IQ3_S");
        e.tok_rb = row_bytes(t->type, D);
        if (dalloc(e, 0, &e.tok_embd, t->nbytes)) return -1;
        CK(cudaMemcpy(e.tok_embd, t->host, t->nbytes, cudaMemcpyHostToDevice));
    }
    if (load_f32(e, 1, "output_hc_norm.weight", HCD, &e.out_norm)) return -1;
    if (load_qw(e, 1, "output_hc_down.weight", e.out_down, HCD, HCR)) return -1;
    if (load_qw(e, 1, "output_hc_up.weight", e.out_up, HCR, HCD, true)) return -1;
    if (load_qw(e, 1, "output.weight", e.head, D, VOCAB)) return -1;
    if (dalloc(e, 1, &e.logits, (size_t) MAXT * VOCAB)) return -1;
    if (dalloc(e, 1, &e.d_argmax, MAXT)) return -1;
    if (dalloc(e, 0, &e.tok_all, e.max_ctx)) return -1;
    if (dalloc(e, 0, &e.d_ids, MAXT)) return -1;
    if (dalloc(e, 0, &e.ple_rows, MAXT * PLE_NH)) return -1;
    if (dalloc(e, 0, &e.ple_emb, MAXT * D)) return -1;
    if (dalloc(e, 0, &e.ple_key, MAXT * HCD)) return -1;
    if (dalloc(e, 0, &e.ple_val, MAXT * D)) return -1;
    if (dalloc(e, 0, &e.ple_gv, MAXT * HCD)) return -1;
    if (dalloc(e, 0, &e.ple_gvn, MAXT * HCD)) return -1;
    CK(cudaSetDevice(0));
    CK(cudaHostAlloc((void **) &e.hbox, (size_t) MAXT * HCD * 4, cudaHostAllocPortable));
    CK(cudaHostAlloc((void **) &e.in_tok, MAXT * sizeof(int), cudaHostAllocPortable));
    CK(cudaHostAlloc((void **) &e.out_am, MAXT * sizeof(int), cudaHostAllocPortable));
    CK(cudaEventCreateWithFlags(&e.ev_handoff, cudaEventDisableTiming));

    // per-layer weights and state
    for (int il = 0; il < NL; ++il) {
        LayerW & L = e.L[il];
        L.dev = il < e.split ? 0 : 1;
        L.attn = is_attn_layer(il);
        L.ple = il == PLE_LAYER;
        const int dv = L.dev;
        const std::string p = "blk." + std::to_string(il) + ".";
        if (load_f32(e, dv, p + "hc_attn_norm.weight", HCD, &L.hca_norm)) return -1;
        if (load_f32(e, dv, p + "hc_ffn_norm.weight", HCD, &L.hcf_norm)) return -1;
        if (load_qw_cat(e, dv, {p + "hc_attn_down.weight", p + "hc_attn_inject.weight"}, {HCR, HC}, L.hca_dinj, HCD)) return -1;
        if (load_qw(e, dv, p + "hc_attn_up.weight", L.hca_up, HCR, HCD, true)) return -1;
        if (load_qw_cat(e, dv, {p + "hc_ffn_down.weight", p + "hc_ffn_inject.weight"}, {HCR, HC}, L.hcf_dinj, HCD)) return -1;
        if (load_qw(e, dv, p + "hc_ffn_up.weight", L.hcf_up, HCR, HCD, true)) return -1;
        if (!L.attn) {
            if (load_qw(e, dv, p + "attn_qkv.weight", L.qkv, D, CONVCH)) return -1;
            if (load_qw(e, dv, p + "attn_gate.weight", L.z, D, DINNER)) return -1;
            if (load_qw_cat(e, dv, {p + "ssm_alpha.weight", p + "ssm_beta.weight"}, {NVH, NVH}, L.ab, D)) return -1;
            if (load_qw(e, dv, p + "ssm_out.weight", L.ssm_out, DINNER, D)) return -1;
            if (load_f32(e, dv, p + "ssm_conv1d.weight", (size_t) CONVCH * 4, &L.conv_w)) return -1;
            if (load_f32(e, dv, p + "ssm_dt.bias", NVH, &L.dt)) return -1;
            if (load_f32(e, dv, p + "ssm_a", NVH, &L.ssm_a)) return -1;
            if (load_f32(e, dv, p + "ssm_norm.weight", DS, &L.ssm_norm)) return -1;
            if (dalloc(e, dv, &L.conv_state, GDN_CST)) return -1;
            if (dalloc(e, dv, &L.S, (size_t) NVH * DS * DS)) return -1;
        } else {
            if (load_qw(e, dv, p + "attn_q.weight", L.q, D, NH * QG)) return -1;
            if (load_qw(e, dv, p + "attn_k.weight", L.k, D, NKV * HD)) return -1;
            if (load_qw(e, dv, p + "attn_v.weight", L.v, D, NKV * HD)) return -1;
            if (load_qw(e, dv, p + "attn_output.weight", L.o, NH * HD, D)) return -1;
            if (load_qw_cat(e, dv, {p + "indexer.q_proj.weight", p + "indexer.k_proj.weight"}, {IXH * IXD, IXD}, L.ixqk, D))
                return -1;
            if (load_f32(e, dv, p + "attn_q_norm.weight", HD, &L.qn)) return -1;
            if (load_f32(e, dv, p + "attn_k_norm.weight", HD, &L.kn)) return -1;
            if (load_f32(e, dv, p + "indexer.q_norm.weight", IXD, &L.ixqn)) return -1;
            if (load_f32(e, dv, p + "indexer.k_norm.weight", IXD, &L.ixkn)) return -1;
            if (dalloc(e, dv, &L.Kc, (size_t) e.max_ctx * NKV * HD)) return -1;
            if (dalloc(e, dv, &L.Vc, (size_t) e.max_ctx * NKV * HD)) return -1;
            if (dalloc(e, dv, &L.IKc, (size_t) e.max_ctx * IXD)) return -1;
            if (dalloc(e, dv, &L.BKc, (size_t) e.maxb * IXD)) return -1;
        }
        if (load_qw_cat(e, dv, {p + "ffn_gate_inp.weight", p + "ffn_gate_inp_shexp.weight"}, {NE, 1}, L.rsg, D)) return -1;
        if (load_qw(e, dv, p + "ffn_gate_shexp.weight", L.sh_gate, D, FFE)) return -1;
        if (load_qw(e, dv, p + "ffn_up_shexp.weight", L.sh_up, D, FFE)) return -1;
        if (load_qw(e, dv, p + "ffn_down_shexp.weight", L.sh_down, FFE, D)) return -1;
        if (load_qw(e, dv, p + "ffn_gate_exps.weight", L.eg, D, NE * FFE)) return -1;
        if (load_qw(e, dv, p + "ffn_up_exps.weight", L.eu, D, NE * FFE)) return -1;
        if (load_qw(e, dv, p + "ffn_down_exps.weight", L.ed, FFE, NE * D)) return -1;
        if (L.eg.type != L.eu.type) FAIL(p + ": gate and up experts differ in type");
        if (L.eg.cv != 256 || (L.ed.type != T_IQ4_NL && L.ed.type != T_Q2_0) || L.ed.cv != 128)
            FAIL(p + ": routed experts in a format this build does not run");
        if (L.ple) {
            if (load_qw(e, dv, p + "ple_key.weight", L.ple_key, D, HCD)) return -1;
            if (load_qw(e, dv, p + "ple_value.weight", L.ple_val, D, D)) return -1;
            if (load_f32(e, dv, p + "ple_norm_key.weight", HCD, &L.ple_nk)) return -1;
            if (load_f32(e, dv, p + "ple_norm_query.weight", HCD, &L.ple_nq)) return -1;
            if (load_f32(e, dv, p + "ple_norm_conv.weight", HCD, &L.ple_nc)) return -1;
            if (load_f32(e, dv, p + "ple_conv1d.weight", (size_t) HCD * 4, &L.ple_conv)) return -1;
            if (dalloc(e, dv, &L.ple_hist, (size_t) PLE_HIST * HCD)) return -1;
        }
    }
    // n-gram table: pinned host RAM, mapped into GPU0's address space
    {
        const GTensor * t = find(e, "per_layer_token_embd.weight");
        if (!t || t->type != T_IQ4_NL || t->ne[0] != PLE_HD) FAIL("per_layer_token_embd: expected IQ4_NL x 160");
        CK(cudaSetDevice(0));
        CK(cudaHostAlloc((void **) &e.ngram, t->nbytes, cudaHostAllocMapped | cudaHostAllocPortable));
        if (read_into(e.g2.hf, t->file_offset, e.ngram, t->nbytes)) return -1;
        const auto & mult = e.g1.iarrays["qwen4exp.ple.layer_multipliers"];
        const auto & size = e.g1.iarrays["qwen4exp.ple.head_vocab_sizes"];
        const auto & off = e.g1.iarrays["qwen4exp.ple.head_offsets"];
        if (mult.size() != 3 || size.size() != PLE_NH || off.size() != PLE_NH) FAIL("bad PLE metadata");
        unsigned long long m[3], sz[PLE_NH], of[PLE_NH];
        for (int i = 0; i < 3; ++i) m[i] = (unsigned long long) mult[i];
        for (int i = 0; i < PLE_NH; ++i) { sz[i] = (unsigned long long) size[i]; of[i] = (unsigned long long) off[i]; }
        CK(cudaMemcpyToSymbol(c_ple_mult, m, sizeof(m)));
        CK(cudaMemcpyToSymbol(c_ple_size, sz, sizeof(sz)));
        CK(cudaMemcpyToSymbol(c_ple_off, of, sizeof(of)));
    }
    for (int dv = 0; dv < 2; ++dv) {
        CK(cudaSetDevice(dv));
        CK(cudaDeviceSynchronize());
        CK(cudaFree(e.stage[dv]));
        e.stage[dv] = nullptr;
    }
    return 0;
}

FN_API int fn_split() { return g_eng ? g_eng->split : -1; }
FN_API double fn_dev_gb(int dv) { return g_eng ? g_eng->bytes_dev[dv] / 1e9 : -1; }
FN_API int fn_n_past() { return g_eng ? g_eng->n_past : -1; }
FN_API double fn_last_ms() { return g_eng ? g_eng->last_ms : -1; }

FN_API int fn_reset() {
    if (!g_eng) FAIL("not loaded");
    Engine & e = *g_eng;
    for (int il = 0; il < NL; ++il) {
        LayerW & L = e.L[il];
        CK(cudaSetDevice(L.dev));
        if (L.conv_state) CK(cudaMemset(L.conv_state, 0, GDN_CST * 4));
        if (L.S) CK(cudaMemset(L.S, 0, (size_t) NVH * DS * DS * 4));
        if (L.ple_hist) CK(cudaMemset(L.ple_hist, 0, (size_t) PLE_HIST * HCD * 4));
    }
    for (int dv = 0; dv < 2; ++dv) { CK(cudaSetDevice(dv)); CK(cudaMemset(e.dv[dv].d_pos, 0, sizeof(int))); }
    e.n_past = 0;
    for (int dv = 0; dv < 2; ++dv) { CK(cudaSetDevice(dv)); CK(cudaDeviceSynchronize()); }
    return 0;
}

// graphs on/off; either way the captured graphs are dropped, so launch settings changed since (fn_moe_override,
// fn_gemv_waves, fn_pdl) apply from the next forward
FN_API int fn_use_graphs(int on) {
    if (!g_eng) FAIL("not loaded");
    for (int dv = 0; dv < 2; ++dv) {
        CK(cudaSetDevice(dv));
        CK(cudaDeviceSynchronize());
        for (auto & ge : g_eng->dv[dv].graph)
            if (ge) { cudaGraphExecDestroy(ge); ge = nullptr; }
    }
    g_eng->use_graphs = on != 0;
    return 0;
}

FN_API int fn_forward(const int * tokens, int nt, float * logits, int * argmax) {
    if (!g_eng) FAIL("not loaded");
    return forward(*g_eng, tokens, nt, logits, argmax);
}

// capture h after every layer of the next forwards into out[NL][nt][HCD] (host memory); null turns it off
FN_API int fn_debug_layers(float * out) {
    if (!g_eng) FAIL("not loaded");
    g_eng->dbg = out;
    return 0;
}

// ---- unit-test hooks (no model needed) ----

template <uint32_t TY> static void dequant_t(const uint8_t * W, uint64_t rb, int rows, int ng, float * out) {
    k_dequant_rows<TY><<<rows, 128>>>(W, rb, nullptr, ng, out);
}

FN_API int fn_test_dequant(uint32_t type, const uint8_t * Wh, int rows, int cols, float * outh) {
    const uint64_t rb = row_bytes(type, cols);
    if (!rb || cols % 32) FAIL("unsupported type/shape");
    CK(cudaSetDevice(0));
    uint8_t * W = nullptr;
    float * out = nullptr;
    CK(cudaMalloc(&W, rb * rows));
    CK(cudaMalloc(&out, (size_t) rows * cols * 4));
    CK(cudaMemcpy(W, Wh, rb * rows, cudaMemcpyHostToDevice));
    const int ng = cols / 32;
    switch (type) {
        case T_BF16: dequant_t<T_BF16>(W, rb, rows, ng, out); break;
        case T_Q8_0: dequant_t<T_Q8_0>(W, rb, rows, ng, out); break;
        case T_Q4_K: dequant_t<T_Q4_K>(W, rb, rows, ng, out); break;
        case T_Q5_K: dequant_t<T_Q5_K>(W, rb, rows, ng, out); break;
        case T_Q6_K: dequant_t<T_Q6_K>(W, rb, rows, ng, out); break;
        case T_IQ2_XXS: dequant_t<T_IQ2_XXS>(W, rb, rows, ng, out); break;
        case T_IQ2_XS: dequant_t<T_IQ2_XS>(W, rb, rows, ng, out); break;
        case T_IQ2_S: dequant_t<T_IQ2_S>(W, rb, rows, ng, out); break;
        case T_IQ3_XXS: dequant_t<T_IQ3_XXS>(W, rb, rows, ng, out); break;
        case T_IQ3_S: dequant_t<T_IQ3_S>(W, rb, rows, ng, out); break;
        case T_IQ4_NL: dequant_t<T_IQ4_NL>(W, rb, rows, ng, out); break;
        case T_IQ4_XS: dequant_t<T_IQ4_XS>(W, rb, rows, ng, out); break;
        case T_Q2_0: dequant_t<T_Q2_0>(W, rb, rows, ng, out); break;
        default: FAIL("unsupported type");
    }
    CK(cudaGetLastError());
    CK(cudaMemcpy(outh, out, (size_t) rows * cols * 4, cudaMemcpyDeviceToHost));
    cudaFree(W);
    cudaFree(out);
    return 0;
}

// benchmarking: force the plan of GEMV weights planned from now on (0 = automatic), the target warps per SM, and the
// routed-expert launch shapes (0 = keep)
FN_API void fn_gemv_override(int wpt, int ks, int tb, int warps_per_sm) {
    g_ov_wpt = wpt;
    g_ov_ks = ks;
    g_ov_tb = tb;
    g_tg_warps = warps_per_sm > 0 ? warps_per_sm : 16;
}

FN_API void fn_gemv_waves(int waves) { g_tg_waves = waves; }

// programmatic dependent launch of the forward kernels on/off (takes effect at the next graph capture)
FN_API void fn_pdl(int on) { g_pdl = on != 0; }
// GEMVs marked early run beside the kernel before them, on/off (A/B; takes effect at the next graph capture)
FN_API void fn_early(int on) { g_early = on != 0; }
FN_API void fn_moe_var(int v) { g_moe_var = v; }

FN_API void fn_moe_override(int gu_wpt, int gu_tb, int dn_wpt, int dn_tb) {
    if (gu_wpt > 0) g_gu_wpt = gu_wpt;
    if (gu_tb > 0) g_gu_tb = gu_tb;
    if (dn_wpt > 0) g_dn_wpt = dn_wpt;
    if (dn_tb > 0) g_dn_tb = dn_tb;
}

// plan of a weight: out = {cv, wpt, ks, tb, nbuf, ntiles, nch}
FN_API int fn_gemv_plan(uint32_t type, int rows, int cols, int * out) {
    QW w;
    if (!plan_qw(w, type, rows, cols)) FAIL("unsupported type/shape");
    const int v[7] = {w.cv, w.wpt, w.ks, w.tb, w.nbuf, w.ntiles, w.nch};
    memcpy(out, v, sizeof(v));
    return 0;
}

// GEMV timing on random bytes of a format, weights cold in L2 (as in a forward, where each weight is read once):
// every call reads its own copy (the 5090's 96 MB L2 would otherwise hold small weights), L2 flushed before timing.
// ms per call

static int g_bench_hot = 0;  // fn_bench_gemv: 1 = one copy of the weight replayed with no L2 flush (weights hot in L2)
FN_API void fn_bench_hot(int on) { g_bench_hot = on; }
FN_API int fn_bench_gemv(uint32_t type, int rows, int cols, int nt, int iters, float * ms) {
    QW w;
    if (nt < 1 || nt > MAXT || !plan_qw(w, type, rows, cols)) FAIL("unsupported type/shape");
    CK(cudaSetDevice(1));
    const size_t tb = tiled_bytes(w);
    float * x = nullptr, * y = nullptr;
    const int copies = g_bench_hot ? 1 : tb * iters <= (2ull << 30) ? iters : std::max<int>(4, (int) ((512ull << 20) / tb + 1));
    uint8_t * flush = nullptr;
    const size_t flush_n = 256ull << 20;
    CK(cudaMalloc(&flush, flush_n));
    CK(cudaMalloc(&w.d, tb * copies));
    CK(cudaMalloc(&w.part, (size_t) TG_PART * 4));
    std::vector<uint8_t> hb(tb);
    for (size_t i = 0; i < hb.size(); ++i) hb[i] = (uint8_t) (i * 2654435761u >> 13);
    if (type == T_BF16 || type == T_Q8_0 || type == T_IQ4_NL || type == T_Q2_0)  // keep scales finite
        for (size_t i = 0; i < hb.size(); i += 2) hb[i + 1] &= 0x3f;
    CK(cudaMemcpy(w.d, hb.data(), hb.size(), cudaMemcpyHostToDevice));
    for (int c = 1; c < copies; ++c) CK(cudaMemcpy(w.d + tb * c, w.d, tb, cudaMemcpyDeviceToDevice));
    CK(cudaMalloc(&x, (size_t) nt * cols * 4));
    CK(cudaMalloc(&y, (size_t) nt * rows * 4));
    CK(cudaMemset(x, 0, (size_t) nt * cols * 4));
    // the calls are replayed as one CUDA graph, as the engine runs them (separate launches measure WDDM launch cost)
    cudaStream_t st;
    CK(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));
    cudaEvent_t a, b;
    cudaEventCreate(&a); cudaEventCreate(&b);
    QW wc = w;
    cudaGraph_t g;
    cudaGraphExec_t ge;
    CK(cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal));
    for (int i = 0; i < iters; ++i) {
        wc.d = w.d + tb * (i % copies);
        gemv(wc, x, cols, y, rows, nt, st);
    }
    CK(cudaStreamEndCapture(st, &g));
    CK(cudaGraphInstantiate(&ge, g, 0));
    CK(cudaGraphLaunch(ge, st));
    if (!g_bench_hot) CK(cudaMemsetAsync(flush, 1, flush_n, st));
    cudaEventRecord(a, st);
    CK(cudaGraphLaunch(ge, st));
    cudaEventRecord(b, st);
    CK(cudaEventSynchronize(b));
    cudaEventElapsedTime(ms, a, b);
    *ms /= iters;
    CK(cudaGetLastError());
    cudaGraphExecDestroy(ge); cudaGraphDestroy(g); cudaStreamDestroy(st); cudaFree(flush);
    cudaFree(w.d); cudaFree(w.part); cudaFree(x); cudaFree(y);
    return 0;
}

// routing on given router outputs part[ks][MAXT][RSG] (split-K partials as the router GEMV leaves them): sel and wsel
// of nt tokens, and the time per call (iters calls replayed as a CUDA graph). variant 0 = k_route_ref, 1 = k_route
FN_API int fn_bench_route(int variant, int nt, int ks, const float * parth, int iters, int * selh, float * wselh, float * ms) {
    CK(cudaSetDevice(1));
    float * part = nullptr, * wsel = nullptr;
    int * sel = nullptr;
    const size_t np = (size_t) ks * MAXT * RSG;
    CK(cudaMalloc(&part, np * 4));
    CK(cudaMalloc(&wsel, MAXT * NEU * 4));
    CK(cudaMalloc(&sel, MAXT * NEU * 4));
    CK(cudaMemcpy(part, parth, np * 4, cudaMemcpyHostToDevice));
    CK(cudaDeviceSynchronize());  // a pageable copy may still be in flight, and st does not wait for the legacy stream
    const GOut lg{part, ks, RSG, (size_t) MAXT * RSG};
    cudaStream_t st;
    CK(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));
    auto run = [&]() {
        if (variant == 0) launch_pdl(k_route_ref, nt, NE, 0, st, lg, sel, wsel);
        else launch_pdl(k_route, nt, NE, 0, st, lg, sel, wsel, lg, lg, (float *) nullptr);
    };
    cudaGraph_t g;
    cudaGraphExec_t ge;
    CK(cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal));
    for (int i = 0; i < iters; ++i) run();
    CK(cudaStreamEndCapture(st, &g));
    CK(cudaGraphInstantiate(&ge, g, 0));
    CK(cudaGraphLaunch(ge, st));
    cudaEvent_t a, b;
    cudaEventCreate(&a); cudaEventCreate(&b);
    cudaEventRecord(a, st);
    CK(cudaGraphLaunch(ge, st));
    cudaEventRecord(b, st);
    CK(cudaEventSynchronize(b));
    cudaEventElapsedTime(ms, a, b);
    *ms /= iters;
    CK(cudaGetLastError());
    CK(cudaMemcpy(selh, sel, nt * NEU * 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(wselh, wsel, nt * NEU * 4, cudaMemcpyDeviceToHost));
    cudaGraphExecDestroy(ge); cudaGraphDestroy(g); cudaStreamDestroy(st);
    cudaFree(part); cudaFree(wsel); cudaFree(sel);
    return 0;
}

// GEMV on GGUF rows Wh (rows x cols of `type`): repacked to the tiled layout, then the engine's gemv
FN_API int fn_test_gemv(uint32_t type, const uint8_t * Wh, int rows, int cols, const float * xh, int nt, float * yh) {
    QW w;
    if (nt < 1 || nt > MAXT || !plan_qw(w, type, rows, cols)) FAIL("unsupported type/shape");
    CK(cudaSetDevice(0));
    uint8_t * raw = nullptr;
    float * x = nullptr, * y = nullptr;
    CK(cudaMalloc(&raw, w.rb * rows));
    CK(cudaMalloc(&w.d, tiled_bytes(w)));
    CK(cudaMalloc(&w.part, (size_t) TG_PART * 4));
    CK(cudaMalloc(&x, (size_t) nt * cols * 4));
    CK(cudaMalloc(&y, (size_t) nt * rows * 4));
    CK(cudaMemcpy(raw, Wh, w.rb * rows, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(x, xh, (size_t) nt * cols * 4, cudaMemcpyHostToDevice));
    repack(w, raw);
    gemv(w, x, cols, y, rows, nt, 0);
    CK(cudaGetLastError());
    CK(cudaMemcpy(yh, y, (size_t) nt * rows * 4, cudaMemcpyDeviceToHost));
    cudaFree(raw); cudaFree(w.d); cudaFree(w.part); cudaFree(x); cudaFree(y);
    return 0;
}
