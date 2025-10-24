"""
#define QK8_1 32
typedef struct {
    GGML_EXTENSION union {
        struct {
            ggml_half d; // delta
            ggml_half s; // d * sum(qs[i])
        } GGML_COMMON_AGGR_S;
        ggml_half2 ds;
    } GGML_COMMON_AGGR_U;
    int8_t qs[QK8_1]; // quants
} block_q8_1;
"""

import torch


def quantize_to_q8_1(input_tensor: torch.Tensor):
    # 确保输入是float16类型
    if input_tensor.dtype != torch.float16:
        input_tensor = input_tensor.to(torch.float16)

    # 展平为一维向量
    flat = input_tensor.flatten()
    n = flat.numel()

    # 检查元素总数是否能被32整除
    if n % 32 != 0:
        raise ValueError("The total number of elements must be divisible by 32.")

    num_groups = n // 32
    # 重塑为 (num_groups, 32) 的矩阵
    groups = flat.reshape(num_groups, 32)

    # 计算每组的最大绝对值
    max_abs = torch.max(torch.abs(groups), dim=1).values  # (num_groups,)

    # 初始化缩放因子d (float16)
    d = torch.zeros(num_groups, dtype=torch.float16, device=input_tensor.device)
    
    # 只对非零最大值组计算d
    non_zero_mask = max_abs != 0
    if non_zero_mask.any():
        d[non_zero_mask] = max_abs[non_zero_mask] / 127.0

    # 创建安全的缩放因子（避免除以0）
    d_safe = d.clone()
    d_safe[d_safe == 0] = 1.0  # 避免除以0

    # 量化：round(value / d) 并裁剪到[-127, 127]
    quantized = torch.round(groups / d_safe.view(-1, 1))
    quantized = torch.clamp(quantized, -127, 127).to(torch.int8)

    # 计算每组量化值的和 (int32避免溢出)
    sum_qs = quantized.sum(dim=1, dtype=torch.int32).to(torch.float16)
    
    # 计算s = d * sum(qs)
    s = d * sum_qs  # (num_groups,)

    # 组合d和s为float16张量 [d, s]
    ds = torch.stack([d, s], dim=1)  # (num_groups, 2)
    
    # 将ds重新解释为int8字节 (4字节)
    ds_bytes = ds.view(torch.int8)  # (num_groups, 4)
    
    # 拼接量化值和缩放因子字节 [4 int8 (d+s) + 32 int8 (qs)]
    quantized_with_ds = torch.cat([ds_bytes, quantized], dim=1)  # (num_groups, 36)
    result = quantized_with_ds.reshape(-1)  # 展平为一维

    return result


def dequantize_q8_1(quantized_tensor: torch.Tensor, original_shape: torch.Size) -> torch.Tensor:
    """
    将Q8_1量化格式的数据反量化为原始float16张量
    
    参数:
        quantized_tensor: 由quantize_to_q8_1产生的量化数据
        original_shape: 原始张量的形状
    
    返回:
        反量化后的float16张量
    """
    # 验证输入类型
    if quantized_tensor.dtype != torch.int8:
        raise ValueError("Quantized tensor must be of type int8")

    # 每组36字节 (4字节d+s + 32字节qs)
    n = quantized_tensor.numel()
    if n % 36 != 0:
        raise ValueError(
            "Invalid quantized tensor size. Expected size divisible by 36 (4 scale bytes + 32 quantized values per group).")

    num_groups = n // 36  # 每组36个元素

    # 重塑为(num_groups, 36)
    quantized_with_ds = quantized_tensor.reshape(num_groups, 36)

    # 分离d+s字节和量化值
    ds_bytes = quantized_with_ds[:, :4]  # (num_groups, 4)
    quantized_values = quantized_with_ds[:, 4:]  # (num_groups, 32)

    # 重新解释为float16 (bitcast)
    ds = ds_bytes.view(torch.float16)  # (num_groups, 2)
    d = ds[:, 0]  # 提取d (num_groups,)
    # s = ds[:, 1]  # s在反量化中不需要

    # 反量化：dequantized = quantized_value * d
    dequantized = quantized_values.to(torch.float16) * d.view(-1, 1)

    # 重组为原始形状
    dequantized_flat = dequantized.reshape(-1)
    dequantized_tensor = dequantized_flat.reshape(original_shape)

    return dequantized_tensor


if __name__ == "__main__":
    mat = torch.randn(2, 16, dtype=torch.float16)
    print("Original matrix:")
    print(mat)
    quant = quantize_to_q8_1(mat)
    print("Quantized tensor:")
    dequant = dequantize_q8_1(quant, mat.shape)
    print("Dequantized matrix:")
    print(dequant)
    print("Difference:")
    print(mat - dequant)
