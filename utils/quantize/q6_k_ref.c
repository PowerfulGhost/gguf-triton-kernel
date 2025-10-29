#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>
#include <math.h>
#include <string.h>
#include <assert.h>

// 在 C 头文件中添加导出声明
#ifdef _WIN32
#define GGML_EXPORT __declspec(dllexport)
#else
#define GGML_EXPORT __attribute__((visibility("default")))
#endif

GGML_EXPORT void quantize_row_q6_K_ref(
    const float *x,
    void *y,
    int64_t k);

#define QK_K 256
#define K_SCALE_SIZE 12
#define GGML_EXTENSION __extension__
#define GROUP_MAX_EPS 1e-15f

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

// 6-bit quantization
// weight is represented as x = a * q
// 16 blocks of 16 elements each
// Effectively 6.5625 bits per weight
typedef struct
{
    uint8_t ql[QK_K / 2];     // quants, lower 4 bits
    uint8_t qh[QK_K / 4];     // quants, upper 2 bits
    int8_t scales[QK_K / 16]; // scales, quantized with 8 bits
    ggml_half d;              // super-block scale
} block_q6_K;

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

static float make_qx_quants(int n, int nmax, const float *GGML_RESTRICT x, int8_t *GGML_RESTRICT L, int rmse_type,
                            const float *GGML_RESTRICT qw)
{
    float max = 0;
    float amax = 0;
    for (int i = 0; i < n; ++i)
    {
        float ax = fabsf(x[i]);
        if (ax > amax)
        {
            amax = ax;
            max = x[i];
        }
    }
    if (amax < GROUP_MAX_EPS)
    { // all zero
        for (int i = 0; i < n; ++i)
        {
            L[i] = 0;
        }
        return 0.f;
    }
    float iscale = -nmax / max;
    if (rmse_type == 0)
    {
        for (int i = 0; i < n; ++i)
        {
            int l = nearest_int(iscale * x[i]);
            L[i] = nmax + MAX(-nmax, MIN(nmax - 1, l));
        }
        return 1 / iscale;
    }
    bool return_early = false;
    if (rmse_type < 0)
    {
        rmse_type = -rmse_type;
        return_early = true;
    }
    float sumlx = 0;
    float suml2 = 0;

    for (int i = 0; i < n; ++i)
    {
        int l = nearest_int(iscale * x[i]);
        l = MAX(-nmax, MIN(nmax - 1, l));
        L[i] = l + nmax;
        float w = qw ? qw[i] : rmse_type == 1 ? x[i] * x[i]
                           : rmse_type == 2   ? 1
                           : rmse_type == 3   ? fabsf(x[i])
                                              : sqrtf(fabsf(x[i]));
        sumlx += w * x[i] * l;
        suml2 += w * l * l;
    }
    float scale = suml2 ? sumlx / suml2 : 0.0f;
    if (return_early)
        return suml2 > 0 ? 0.5f * (scale + 1 / iscale) : 1 / iscale;
    float best = scale * sumlx;
    for (int is = -9; is <= 9; ++is)
    {
        if (is == 0)
        {
            continue;
        }
        iscale = -(nmax + 0.1f * is) / max;
        sumlx = suml2 = 0;
        for (int i = 0; i < n; ++i)
        {
            int l = nearest_int(iscale * x[i]);
            l = MAX(-nmax, MIN(nmax - 1, l));
            float w = qw ? qw[i] : rmse_type == 1 ? x[i] * x[i]
                               : rmse_type == 2   ? 1
                               : rmse_type == 3   ? fabsf(x[i])
                                                  : sqrtf(fabsf(x[i]));
            sumlx += w * x[i] * l;
            suml2 += w * l * l;
        }
        if (suml2 > 0 && sumlx * sumlx > best * suml2)
        {
            for (int i = 0; i < n; ++i)
            {
                int l = nearest_int(iscale * x[i]);
                L[i] = nmax + MAX(-nmax, MIN(nmax - 1, l));
            }
            scale = sumlx / suml2;
            best = scale * sumlx;
        }
    }
    return scale;
}

void quantize_row_q6_K_ref(const float *GGML_RESTRICT x, void *GGML_RESTRICT y_void, int64_t k)
{
    block_q6_K *y = (block_q6_K *)y_void;
    assert(k % QK_K == 0);
    const int64_t nb = k / QK_K;

    int8_t L[QK_K];
    float scales[QK_K / 16];

    for (int i = 0; i < nb; i++)
    {

        float max_scale = 0;
        float max_abs_scale = 0;

        for (int ib = 0; ib < QK_K / 16; ++ib)
        {

            const float scale = make_qx_quants(16, 32, x + 16 * ib, L + 16 * ib, 1, NULL);
            scales[ib] = scale;

            const float abs_scale = fabsf(scale);
            if (abs_scale > max_abs_scale)
            {
                max_abs_scale = abs_scale;
                max_scale = scale;
            }
        }

        if (max_abs_scale < GROUP_MAX_EPS)
        {
            memset(&y[i], 0, sizeof(block_q6_K));
            y[i].d = GGML_FP32_TO_FP16(0.f);
            x += QK_K;
            continue;
        }

        float iscale = -128.f / max_scale;
        y[i].d = GGML_FP32_TO_FP16(1 / iscale);
        for (int ib = 0; ib < QK_K / 16; ++ib)
        {
            y[i].scales[ib] = MIN(127, nearest_int(iscale * scales[ib]));
        }

        for (int j = 0; j < QK_K / 16; ++j)
        {
            float d = GGML_FP16_TO_FP32(y[i].d) * y[i].scales[j];
            if (!d)
            {
                continue;
            }
            for (int ii = 0; ii < 16; ++ii)
            {
                int l = nearest_int(x[16 * j + ii] / d);
                l = MAX(-32, MIN(31, l));
                L[16 * j + ii] = l + 32;
            }
        }

        // printf("y[i].d: %f\n", GGML_FP16_TO_FP32(y[i].d));
        // printf("scales: ");
        // for (int ii = 0; ii < 16; ii++)
        // {
        //     printf("%i ", y[i].scales[ii]);
        // }
        // printf("\n");
        // for (int kk = 0; kk < 16; kk++)
        // {
        //     printf("L[%i:%i]:\n[", kk, kk + 16);
        //     for (int ii = 0; ii < 16; ii++)
        //     {
        //         printf("%i", L[16*kk + ii]);
        //         if (ii != 15) printf(" ,");
        //     }
        //     printf("]\n");
        // }

        uint8_t *GGML_RESTRICT ql = y[i].ql;
        uint8_t *GGML_RESTRICT qh = y[i].qh;
        for (int j = 0; j < QK_K; j += 128)
        {
            for (int l = 0; l < 32; ++l)
            {
                const uint8_t q1 = L[j + l + 0] & 0xF;
                const uint8_t q2 = L[j + l + 32] & 0xF;
                const uint8_t q3 = L[j + l + 64] & 0xF;
                const uint8_t q4 = L[j + l + 96] & 0xF;
                ql[l + 0] = q1 | (q3 << 4);
                ql[l + 32] = q2 | (q4 << 4);
                qh[l] = (L[j + l] >> 4) | ((L[j + l + 32] >> 4) << 2) | ((L[j + l + 64] >> 4) << 4) | ((L[j + l + 96] >> 4) << 6);
            }
            ql += 64;
            qh += 32;
        }

        x += QK_K;
    }
}
