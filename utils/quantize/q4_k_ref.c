/*
Copied and modified from GGML
for utility usage only
*/

#include <stdint.h>
#include <stdbool.h>
#include <math.h>
#include <string.h>
#include <assert.h>

#ifdef _WIN32
#define GGML_EXPORT __declspec(dllexport)
#else
#define GGML_EXPORT __attribute__((visibility("default")))
#endif

GGML_EXPORT void quantize_row_q4_K_ref(
    const float *x,
    void *y,
    int64_t k);

#define QK_K 256
#define K_SCALE_SIZE 12
#define GGML_EXTENSION __extension__

// std-c++ allow anonymous unions but some compiler warn on it
#define GGML_COMMON_AGGR_U data
// std-c++ do not allow it.
#define GGML_COMMON_AGGR_S data

#ifdef __cplusplus
// restrict not standard in C++
#if defined(__GNUC__)
#define GGML_RESTRICT __restrict__
#elif defined(__clang__)
#define GGML_RESTRICT __restrict
#elif defined(_MSC_VER)
#define GGML_RESTRICT __restrict
#else
#define GGML_RESTRICT
#endif
#else
#if defined(_MSC_VER) && (__STDC_VERSION__ < 201112L)
#define GGML_RESTRICT __restrict
#else
#define GGML_RESTRICT restrict
#endif
#endif

#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define GGML_FP16_TO_FP32(x) ggml_compute_fp16_to_fp32(x)
#define GGML_FP32_TO_FP16(x) ggml_compute_fp32_to_fp16(x)

typedef uint16_t ggml_half;
typedef uint16_t ggml_fp16_t;
typedef uint32_t ggml_half2;

// 4-bit quantization
// 8 blocks of 32 elements each
// weight is represented as x = a * q + b
// Effectively 4.5 bits per weight
// typedef struct {
//     GGML_EXTENSION union {
//         struct {
//             ggml_half d;    // super-block scale for quantized scales
//             ggml_half dmin; // super-block scale for quantized mins
//         } GGML_COMMON_AGGR_S;
//         ggml_half2 dm;
//     } GGML_COMMON_AGGR_U;
//     uint8_t scales[K_SCALE_SIZE]; // scales and mins, quantized with 6 bits
//     uint8_t qs[QK_K/2];           // 4--bit quants
// } block_q4_K;

typedef struct
{
    union
    {
        struct
        {
            ggml_half d;    // super-block scale for quantized scales
            ggml_half dmin; // super-block scale for quantized mins
        };
        ggml_half2 dm;
    };
    uint8_t scales[K_SCALE_SIZE]; // scales and mins, quantized with 6 bits
    uint8_t qs[QK_K / 2];         // 4--bit quants
} block_q4_K;

static inline float fp32_from_bits(uint32_t w)
{
    union
    {
        uint32_t as_bits;
        float as_value;
    } fp32;
    fp32.as_bits = w;
    return fp32.as_value;
}

static inline uint32_t fp32_to_bits(float f)
{
    union
    {
        float as_value;
        uint32_t as_bits;
    } fp32;
    fp32.as_value = f;
    return fp32.as_bits;
}

static inline float ggml_compute_fp16_to_fp32(ggml_fp16_t h)
{
    const uint32_t w = (uint32_t)h << 16;
    const uint32_t sign = w & UINT32_C(0x80000000);
    const uint32_t two_w = w + w;

    const uint32_t exp_offset = UINT32_C(0xE0) << 23;
#if (defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 199901L) || defined(__GNUC__) && !defined(__STRICT_ANSI__)) && (!defined(__cplusplus) || __cplusplus >= 201703L)
    const float exp_scale = 0x1.0p-112f;
#else
    const float exp_scale = fp32_from_bits(UINT32_C(0x7800000));
#endif
    const float normalized_value = fp32_from_bits((two_w >> 4) + exp_offset) * exp_scale;

    const uint32_t magic_mask = UINT32_C(126) << 23;
    const float magic_bias = 0.5f;
    const float denormalized_value = fp32_from_bits((two_w >> 17) | magic_mask) - magic_bias;

    const uint32_t denormalized_cutoff = UINT32_C(1) << 27;
    const uint32_t result = sign |
                            (two_w < denormalized_cutoff ? fp32_to_bits(denormalized_value) : fp32_to_bits(normalized_value));
    return fp32_from_bits(result);
}

static inline ggml_fp16_t ggml_compute_fp32_to_fp16(float f)
{
#if (defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 199901L) || defined(__GNUC__) && !defined(__STRICT_ANSI__)) && (!defined(__cplusplus) || __cplusplus >= 201703L)
    const float scale_to_inf = 0x1.0p+112f;
    const float scale_to_zero = 0x1.0p-110f;
#else
    const float scale_to_inf = fp32_from_bits(UINT32_C(0x77800000));
    const float scale_to_zero = fp32_from_bits(UINT32_C(0x08800000));
#endif
    float base = (fabsf(f) * scale_to_inf) * scale_to_zero;

    const uint32_t w = fp32_to_bits(f);
    const uint32_t shl1_w = w + w;
    const uint32_t sign = w & UINT32_C(0x80000000);
    uint32_t bias = shl1_w & UINT32_C(0xFF000000);
    if (bias < UINT32_C(0x71000000))
    {
        bias = UINT32_C(0x71000000);
    }

    base = fp32_from_bits((bias >> 1) + UINT32_C(0x07800000)) + base;
    const uint32_t bits = fp32_to_bits(base);
    const uint32_t exp_bits = (bits >> 13) & UINT32_C(0x00007C00);
    const uint32_t mantissa_bits = bits & UINT32_C(0x00000FFF);
    const uint32_t nonsign = exp_bits + mantissa_bits;
    return (sign >> 16) | (shl1_w > UINT32_C(0xFF000000) ? UINT16_C(0x7E00) : nonsign);
}

static inline int nearest_int(float fval)
{
    assert(fabsf(fval) <= 4194303.f);
    float val = fval + 12582912.f;
    int i;
    memcpy(&i, &val, sizeof(int));
    return (i & 0x007fffff) - 0x00400000;
}

static inline void get_scale_min_k4(int j, const uint8_t *GGML_RESTRICT q, uint8_t *GGML_RESTRICT d, uint8_t *GGML_RESTRICT m)
{
    if (j < 4)
    {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    }
    else
    {
        *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4);
    }
}

static float make_qkx2_quants(int n, int nmax, const float *GGML_RESTRICT x, const float *GGML_RESTRICT weights,
                              uint8_t *GGML_RESTRICT L, float *GGML_RESTRICT the_min, uint8_t *GGML_RESTRICT Laux,
                              float rmin, float rdelta, int nstep, bool use_mad)
{
    float min = x[0];
    float max = x[0];
    float sum_w = weights[0];
    float sum_x = sum_w * x[0];

    for (int i = 1; i < n; ++i)
    {
        if (x[i] < min)
            min = x[i];
        if (x[i] > max)
            max = x[i];
        float w = weights[i];
        sum_w += w;
        sum_x += w * x[i];
    }
    if (min > 0)
        min = 0;
    if (max == min)
    {
        for (int i = 0; i < n; ++i)
            L[i] = 0;
        *the_min = -min;
        return 0.f;
    }
    float iscale = nmax / (max - min);
    float scale = 1 / iscale;
    float best_error = 0;
    for (int i = 0; i < n; ++i)
    {
        int l = nearest_int(iscale * (x[i] - min));
        L[i] = MAX(0, MIN(nmax, l));
        float diff = scale * L[i] + min - x[i];
        diff = use_mad ? fabsf(diff) : diff * diff;
        float w = weights[i];
        best_error += w * diff;
    }
    if (nstep < 1)
    {
        *the_min = -min;
        return scale;
    }
    for (int is = 0; is <= nstep; ++is)
    {
        iscale = (rmin + rdelta * is + nmax) / (max - min);
        float sum_l = 0, sum_l2 = 0, sum_xl = 0;
        for (int i = 0; i < n; ++i)
        {
            int l = nearest_int(iscale * (x[i] - min));
            l = MAX(0, MIN(nmax, l));
            Laux[i] = l;
            float w = weights[i];
            sum_l += w * l;
            sum_l2 += w * l * l;
            sum_xl += w * l * x[i];
        }
        float D = sum_w * sum_l2 - sum_l * sum_l;
        if (D > 0)
        {
            float this_scale = (sum_w * sum_xl - sum_x * sum_l) / D;
            float this_min = (sum_l2 * sum_x - sum_l * sum_xl) / D;
            if (this_min > 0)
            {
                this_min = 0;
                this_scale = sum_xl / sum_l2;
            }
            float cur_error = 0;
            for (int i = 0; i < n; ++i)
            {
                float diff = this_scale * Laux[i] + this_min - x[i];
                diff = use_mad ? fabsf(diff) : diff * diff;
                float w = weights[i];
                cur_error += w * diff;
            }
            if (cur_error < best_error)
            {
                for (int i = 0; i < n; ++i)
                {
                    L[i] = Laux[i];
                }
                best_error = cur_error;
                scale = this_scale;
                min = this_min;
            }
        }
    }
    *the_min = -min;
    return scale;
}

void quantize_row_q4_K_ref(const float *GGML_RESTRICT x, void *GGML_RESTRICT y_void, int64_t k)
{
    block_q4_K *y = (block_q4_K *)y_void;
    assert(k % QK_K == 0);
    const int nb = k / QK_K;

    uint8_t L[QK_K];
    uint8_t Laux[32];
    float weights[32];
    float mins[QK_K / 32];
    float scales[QK_K / 32];

    for (int i = 0; i < nb; i++)
    {
        float max_scale = 0; // as we are deducting the min, scales are always positive
        float max_min = 0;
        for (int j = 0; j < QK_K / 32; ++j)
        {
            // scales[j] = make_qkx1_quants(32, 15, x + 32*j, L + 32*j, &mins[j], 9, 0.5f);
            float sum_x2 = 0;
            for (int l = 0; l < 32; ++l)
                sum_x2 += x[32 * j + l] * x[32 * j + l];
            float av_x = sqrtf(sum_x2 / 32);
            for (int l = 0; l < 32; ++l)
                weights[l] = av_x + fabsf(x[32 * j + l]);
            scales[j] = make_qkx2_quants(32, 15, x + 32 * j, weights, L + 32 * j, &mins[j], Laux, -1.f, 0.1f, 20, false);
            float scale = scales[j];
            if (scale > max_scale)
            {
                max_scale = scale;
            }
            float min = mins[j];
            if (min > max_min)
            {
                max_min = min;
            }
        }

        float inv_scale = max_scale > 0 ? 63.f / max_scale : 0.f;
        float inv_min = max_min > 0 ? 63.f / max_min : 0.f;
        for (int j = 0; j < QK_K / 32; ++j)
        {
            uint8_t ls = nearest_int(inv_scale * scales[j]);
            uint8_t lm = nearest_int(inv_min * mins[j]);
            ls = MIN(63, ls);
            lm = MIN(63, lm);
            if (j < 4)
            {
                y[i].scales[j] = ls;
                y[i].scales[j + 4] = lm;
            }
            else
            {
                y[i].scales[j + 4] = (ls & 0xF) | ((lm & 0xF) << 4);
                y[i].scales[j - 4] |= ((ls >> 4) << 6);
                y[i].scales[j - 0] |= ((lm >> 4) << 6);
            }
        }
        y[i].d = GGML_FP32_TO_FP16(max_scale / 63.f);
        y[i].dmin = GGML_FP32_TO_FP16(max_min / 63.f);

        uint8_t sc, m;
        for (int j = 0; j < QK_K / 32; ++j)
        {
            get_scale_min_k4(j, y[i].scales, &sc, &m);
            const float d = GGML_FP16_TO_FP32(y[i].d) * sc;
            if (!d)
                continue;
            const float dm = GGML_FP16_TO_FP32(y[i].dmin) * m;
            for (int ii = 0; ii < 32; ++ii)
            {
                int l = nearest_int((x[32 * j + ii] + dm) / d);
                l = MAX(0, MIN(15, l));
                L[32 * j + ii] = l;
            }
        }

        uint8_t *q = y[i].qs;
        for (int j = 0; j < QK_K; j += 64)
        {
            for (int l = 0; l < 32; ++l)
                q[l] = L[j + l] | (L[j + l + 32] << 4);
            q += 32;
        }

        x += QK_K;
    }
}
