import torch


def quantize_to_q8_0(input_tensor: torch.Tensor):
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
    max_vals = torch.max(torch.abs(groups), dim=1).values  # (num_groups,)

    # 创建缩放因子张量 (float16)
    scales = torch.ones(num_groups, dtype=torch.float16)
    non_zero_mask = max_vals != 0

    # 只对非零最大值组计算缩放因子
    if non_zero_mask.any():
        scale_factor = torch.tensor(127.0, dtype=torch.float16)
        scales[non_zero_mask] = max_vals[non_zero_mask] / scale_factor

    # 量化：round(value / scale) 并裁剪到[-127, 127]
    quantized = torch.round(groups / scales.view(-1, 1))
    quantized = torch.clamp(quantized, -127, 127).to(torch.int8)

    # 修复：正确处理缩放因子的bit cast
    # 1. 确保scales是连续的
    scales = scales.contiguous()
    # 2. 直接将float16张量view为int8（每个float16变成2个int8）
    scales_bytes = scales.view(torch.int8)
    # 3. 重塑为(num_groups, 2)
    scales_bytes = scales_bytes.reshape(num_groups, 2)

    # 拼接量化值和缩放因子字节 [2 int8 + 32 int8]
    quantized_with_scales = torch.cat([scales_bytes, quantized], dim=1)
    result = quantized_with_scales.reshape(-1)  # 展平为一维

    return result


def dequantize_q8_0(quantized_tensor: torch.Tensor, original_shape: torch.Size) -> torch.Tensor:
    """
    将Q8_0量化格式的数据反量化为原始float16张量
    
    参数:
        quantized_tensor: 由quantize_to_q8_0产生的量化数据
        original_shape: 原始张量的形状
    
    返回:
        反量化后的float16张量
    """
    # 验证输入类型
    if quantized_tensor.dtype != torch.int8:
        raise ValueError("Quantized tensor must be of type int8")

    # 计算组数
    n = quantized_tensor.numel()
    if n % 34 != 0:
        raise ValueError(
            "Invalid quantized tensor size. Expected size divisible by 34 (2 scale bytes + 32 quantized values per group).")

    num_groups = n // 34  # 每组34个元素 (2个缩放因子字节 + 32个量化值)

    # 重塑为(num_groups, 34)
    quantized_with_scales = quantized_tensor.reshape(num_groups, 34)

    # 分离缩放因子字节和量化值
    scales_bytes = quantized_with_scales[:, :2]  # (num_groups, 2)
    quantized_values = quantized_with_scales[:, 2:]  # (num_groups, 32)

    # 重新解释为float16 (bitcast，保持二进制不变)
    # 1. 确保连续
    scales_bytes = scales_bytes.contiguous()
    # 2. 将int8重新解释为uint8 (bitcast，保持二进制不变)
    scales_bytes_uint8 = scales_bytes.view(torch.uint8)
    # 3. 将uint8重新解释为float16
    scales = scales_bytes_uint8.view(torch.float16)

    # 现在scales是(num_groups,)的float16张量
    scales = scales.reshape(num_groups)

    # 反量化：dequantized = quantized_value * scale
    dequantized = quantized_values.to(torch.float16) * scales.view(-1, 1)

    # 重组为原始形状
    dequantized_flat = dequantized.reshape(-1)
    dequantized_tensor = dequantized_flat.reshape(original_shape)

    return dequantized_tensor


if __name__ == "__main__":
    mat = torch.randn(2, 16, dtype=torch.float16)
    print("Original matrix:")
    print(mat)
    quant = quantize_to_q8_0(mat)
    print("Quantized tensor:")
    dequant = dequantize_q8_0(quant, mat.shape)
    print("Dequantized matrix:")
    print(dequant)
    print("Difference:")
    print(mat - dequant)
