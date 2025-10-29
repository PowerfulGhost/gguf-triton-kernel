"""
// 6-bit quantization
// weight is represented as x = a * q
// 16 blocks of 16 elements each
// Effectively 6.5625 bits per weight
#define QK_K 256
#define K_SCALE_SIZE 12
typedef struct {
    uint8_t ql[QK_K/2];      // quants, lower 4 bits
    uint8_t qh[QK_K/4];      // quants, upper 2 bits
    int8_t  scales[QK_K/16]; // scales, quantized with 8 bits
    ggml_half d;             // super-block scale
} block_q6_K;

#define QK8_0 32
typedef struct {
    ggml_half d;       // delta
    int8_t  qs[QK8_0]; // quants
} block_q8_0;
"""

import os
import torch
import numpy as np
import ctypes
from ctypes import c_uint16, c_uint8, c_int8, c_void_p, c_int64, POINTER, Structure, byref


# 定义 block_q6_K 结构体 (210 字节)
class BlockQ6K(Structure):
    _pack_ = 1
    _fields_ = [
        ("ql", c_uint8 * 128),  # QK_K/2 = 128 bytes
        ("qh", c_uint8 * 64),  # QK_K/4 = 64 bytes
        ("scales", c_int8 * 16),  # QK_K/16 = 16 bytes
        ("d", c_uint16)  # ggml_half (2 bytes)
    ]


# 创建匹配的 NumPy dtype
BLOCK_DTYPE_Q6K = np.dtype([
    ('ql', 'u1', (128, )),
    ('qh', 'u1', (64, )),
    ('scales', 'i1', (16, )),
    ('d', '<u2')  # 小端序 uint16 存储半精度浮点数
])

# 加载共享库
current_dir = os.path.dirname(os.path.abspath(__file__))
so_path = os.path.join(current_dir, "libq6_k_ref.so")
if not os.path.exists(so_path):
    raise FileNotFoundError(f"""{so_path} not found!
complie it with:
cd {current_dir}
gcc -shared -fPIC -o {so_path} {current_dir}/q6_k_ref.c""")  # 修正编译命令: -fPIO -> -fPIC
lib = ctypes.CDLL(so_path)

# 设置函数参数和返回类型
lib.quantize_row_q6_K_ref.argtypes = [
    POINTER(ctypes.c_float),  # float* x
    c_void_p,  # void* y
    c_int64  # int64_t k
]
lib.quantize_row_q6_K_ref.restype = None

# 验证结构体大小
assert ctypes.sizeof(BlockQ6K) == 210, f"Struct size mismatch! Expected 210, got {ctypes.sizeof(BlockQ6K)}"


def _quantize_q6k_np(arr: np.ndarray) -> np.ndarray:
    """
    将 float32 数组量化为 q6_K 格式
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
    output_ctypes = (BlockQ6K * num_blocks)()

    # 调用 C 函数
    lib.quantize_row_q6_K_ref(arr.ctypes.data_as(POINTER(ctypes.c_float)), byref(output_ctypes), c_int64(n))

    # 转换为 NumPy 结构化数组 (保持零拷贝)
    return np.frombuffer(output_ctypes, dtype=BLOCK_DTYPE_Q6K, count=num_blocks)


def quantize_to_q6_k(input_tensor: torch.Tensor) -> torch.Tensor:
    """
    将 PyTorch 张量量化为 q6_K 格式
    :param input_tensor: 输入张量 (float32)
    :return: 量化后的张量 (int8 类型, 形状为 [n_blocks, 210])
    """
    arr = input_tensor.cpu().to(torch.float32).numpy()
    # 转换为结构化数组后展平为字节流
    structured = _quantize_q6k_np(arr)
    byte_view = structured.view(dtype=np.uint8)
    ret_torch = torch.tensor(byte_view, dtype=torch.int8)
    return ret_torch


# =============================== 反量化实现 ===============================

QK_K = 256  # 每个 block 的元素数量
K_SCALE_SIZE = 16  # scales 数组的大小 (QK_K/16)


def dequantize_blocks(blocks: np.ndarray) -> np.ndarray:
    n_blocks = blocks.shape[0]

    ql, rest = np.hsplit(blocks, [QK_K // 2])
    qh, rest = np.hsplit(rest, [QK_K // 4])
    scales, d = np.hsplit(rest, [QK_K // 16])

    scales = scales.view(np.int8).astype(np.float32)
    d = d.view(np.float16).astype(np.float32)
    d = (d * scales).reshape((n_blocks, QK_K // 16, 1))

    ql = ql.reshape((n_blocks, -1, 1, 64)) >> np.array([0, 4], dtype=np.uint8).reshape((1, 1, 2, 1))
    ql = (ql & np.uint8(0x0F)).reshape((n_blocks, -1, 32))
    qh = qh.reshape((n_blocks, -1, 1, 32)) >> np.array([0, 2, 4, 6], dtype=np.uint8).reshape((1, 1, 4, 1))
    qh = (qh & np.uint8(0x03)).reshape((n_blocks, -1, 32))
    q = (ql | (qh << np.uint8(4))).astype(np.int8) - np.int8(32)
    q = q.reshape((n_blocks, QK_K // 16, -1)).astype(np.float32)

    return (d * q).reshape((n_blocks, QK_K))


def dequantize_q6_k(quantized_tensor: torch.Tensor, original_shape: torch.Size) -> torch.Tensor:
    """
    将 q6_K 量化张量反量化为原始浮点张量
    :param quantized_tensor: 量化后的张量 (int8 类型, 形状为 [n_blocks * 210])
    :param original_shape: 原始张量的形状
    :return: 反量化后的浮点张量
    """
    n = quantized_tensor.numel()
    if n % 210 != 0:
        raise ValueError(f"Invalid quantized tensor size. Expected size divisible by 210, got {n}.")

    n_blocks = n // 210
    # 转换为字节数组并重塑为块结构
    quantized_np = quantized_tensor.cpu().numpy().reshape((n_blocks, 210))

    # 反量化
    dequantized_np = dequantize_blocks(quantized_np)

    # 恢复原始形状
    dequantized_tensor = torch.tensor(dequantized_np, dtype=torch.float32).reshape(original_shape)

    return dequantized_tensor


# ===============================

# 使用示例
if __name__ == "__main__":
    torch.manual_seed(42)

    M, K = 1, 256
    A = torch.randn((M, K), dtype=torch.float16)

    print(f"{A=}")

    quant_A = quantize_to_q6_k(A)
    print(f"{quant_A=}")

    dequant_A = dequantize_q6_k(quant_A, (M, K))
    print(f"{dequant_A=}")

    diff = A - dequant_A
    print(f"{diff=}")

    percent = torch.abs(diff) / torch.abs(A) * 100
    torch.set_printoptions(precision=1, sci_mode=False)
    print(f"{percent=}")
