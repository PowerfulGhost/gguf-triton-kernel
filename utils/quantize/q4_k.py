"""
// 4-bit quantization
// 8 blocks of 32 elements each
// weight is represented as x = a * q + b
// Effectively 4.5 bits per weight
#define QK_K 256
#define K_SCALE_SIZE 12
typedef struct {
    GGML_EXTENSION union {
        struct {
            ggml_half d;    // super-block scale for quantized scales
            ggml_half dmin; // super-block scale for quantized mins
        } GGML_COMMON_AGGR_S;
        ggml_half2 dm;
    } GGML_COMMON_AGGR_U;
    uint8_t scales[K_SCALE_SIZE]; // scales and mins, quantized with 6 bits
    uint8_t qs[QK_K/2];           // 4--bit quants
} block_q4_K;
"""


import os
import torch
import numpy as np
import ctypes
from ctypes import c_uint16, c_uint8, c_void_p, c_int64, POINTER, Structure, byref


# 定义 block_q4_K 结构体 (144 字节)
class BlockQ4K(Structure):
    _pack_ = 1
    _fields_ = [("d", c_uint16), ("dmin", c_uint16), ("scales", c_uint8 * 12), ("qs", c_uint8 * 128)]


# 创建匹配的 NumPy dtype
BLOCK_DTYPE = np.dtype([('d', '<u2'), ('dmin', '<u2'), ('scales', 'u1', (12, )), ('qs', 'u1', (128, ))])

# 加载共享库
current_dir = os.path.dirname(os.path.abspath(__file__))
so_path = os.path.join(current_dir, "libq4_k_ref.so")
if not os.path.exists(so_path):
    raise FileNotFoundError(f"""{so_path} not found!
complie it with:
cd {current_dir}
gcc -shared -fPIO -o {so_path} {current_dir}/q4_k_ref.c""")
lib = ctypes.CDLL(so_path)

# 设置函数参数和返回类型
lib.quantize_row_q4_K_ref.argtypes = [
    POINTER(ctypes.c_float),  # float* x
    c_void_p,  # void* y
    c_int64  # int64_t k
]
lib.quantize_row_q4_K_ref.restype = None

# 验证结构体大小
assert ctypes.sizeof(BlockQ4K) == 144, f"Struct size mismatch! Expected 144, got {ctypes.sizeof(BlockQ4K)}"


def _quantize_q4k_np(arr: np.ndarray) -> np.ndarray:
    """
    将 float32 数组量化为 q4_K 格式
    :param arr: 输入数组 (必须是 C 连续的 float32, 长度需为 256 的倍数)
    :return: 量化后的结构体数组 (NumPy 结构化数组)
    """
    # 验证输入
    if arr.dtype != np.float32:
        raise TypeError("Input must be float32 array")
    if not arr.flags['C_CONTIGUOUS']:
        arr = np.ascontiguousarray(arr)
    n = arr.size
    if n % 256 != 0:
        raise ValueError(f"Array length must be multiple of 256 (got {n})")

    num_blocks = n // 256

    # 创建输出缓冲区 (作为 ctypes 数组)
    output_ctypes = (BlockQ4K * num_blocks)()

    # 调用 C 函数
    lib.quantize_row_q4_K_ref(arr.ctypes.data_as(POINTER(ctypes.c_float)), byref(output_ctypes), c_int64(n))

    # 转换为 NumPy 结构化数组 (保持零拷贝)
    return np.frombuffer(output_ctypes, dtype=BLOCK_DTYPE, count=num_blocks)


def quantize_to_q4_k(input_tensor: torch.Tensor) -> torch.Tensor:
    arr = input_tensor.cpu().to(torch.float32).numpy()
    ret = _quantize_q4k_np(arr).view(dtype=np.int8)
    ret_torch = torch.tensor(ret)
    return ret_torch


# ===============================

QK_K = 256
K_SCALE_SIZE = 12


def get_scale_min(scales: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n_blocks = scales.shape[0]
    scales = scales.view(np.uint8)
    ### Unpacking the following: ###
    #  0 EEAAAAAA
    #  1 FFBBBBBB
    #  2 GGCCCCCC
    #  3 HHDDDDDD
    #  4 eeaaaaaa
    #  5 ffbbbbbb
    #  6 ggcccccc
    #  7 hhdddddd
    #  8 eeeeEEEE
    #  9 ffffFFFF
    # 10 ggggGGGG
    # 11 hhhhHHHH
    scales = scales.reshape((n_blocks, 3, 4))
    d, m, m_d = np.split(scales, 3, axis=-2)

    sc = np.concatenate([d & 0x3F, (m_d & 0x0F) | ((d >> 2) & 0x30)], axis=-1)
    min = np.concatenate([m & 0x3F, (m_d >> 4) | ((m >> 2) & 0x30)], axis=-1)

    return (sc.reshape((n_blocks, 8)), min.reshape((n_blocks, 8)))


def dequantize_blocks(blocks: np.ndarray) -> np.ndarray:
    n_blocks = blocks.shape[0]

    d, rest = np.hsplit(blocks, [2])
    dmin, rest = np.hsplit(rest, [2])
    scales, qs = np.hsplit(rest, [12])

    d = d.view(np.float16).astype(np.float32)
    dmin = dmin.view(np.float16).astype(np.float32)

    sc, m = get_scale_min(scales)

    d = (d * sc.astype(np.float32)).reshape((n_blocks, -1, 1))
    dm = (dmin * m.astype(np.float32)).reshape((n_blocks, -1, 1))

    qs = qs.reshape((n_blocks, -1, 1, 32)) >> np.array([0, 4], dtype=np.uint8).reshape((1, 1, 2, 1))
    qs = (qs & np.uint8(0x0F)).reshape((n_blocks, -1, 32)).astype(np.float32)

    return (d * qs - dm).reshape((n_blocks, QK_K))


def dequantize_q4_k(quantized_tensor: torch.Tensor, original_shape: torch.Size) -> torch.Tensor:
    n = quantized_tensor.numel()
    if n % 144 != 0:
        raise ValueError(f"Invalid quantized tensor size. Expected size divisible by 144, got {n}.")

    n_super_blocks = n // 144
    quantized_tensor_np = quantized_tensor.numpy().reshape((n_super_blocks, -1))

    dequantized_tensor_np = dequantize_blocks(quantized_tensor_np)

    dequantized_tensor = torch.tensor(dequantized_tensor_np).to(torch.float16).reshape(original_shape)

    return dequantized_tensor


# ===============================

# 使用示例
if __name__ == "__main__":
    M, K = 1, 256
    A = torch.randn((M, K), dtype=torch.float16)

    print(f"{A=}")

    quant_A = quantize_to_q4_k(A)
    print(f"{quant_A=}")

    dequant_A = dequantize_q4_k(quant_A, (M, K))
    print(f"{dequant_A=}")

    diff = A - dequant_A
    print(f"{diff=}")

    percent = torch.abs(diff) / torch.abs(A) * 100
    torch.set_printoptions(precision=1, sci_mode=False)
    print(f"{percent=}")
