import triton
import triton.language as tl
import torch


@triton.jit
def load_int8x2_as_fp16(ptr, mask):
    lower_byte = tl.load(ptr, mask).cast(tl.uint8, bitcast=True)
    higher_byte = tl.load(ptr + 1, mask).cast(tl.uint8, bitcast=True)

    result = higher_byte.cast(tl.uint16) << 8 | lower_byte.cast(tl.uint16)

    return result.cast(tl.float16, bitcast=True)


@triton.jit
def mmq_q8_0_kernel(A_ptr, B_ptr, C_ptr, M, N, K, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    # Block IDs
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Block starting indices
    m0 = pid_m * BLOCK_M
    n0 = pid_n * BLOCK_N

    # Thread indices within block
    tid = tl.arange(0, BLOCK_M * BLOCK_N)
    tid_m = tid // BLOCK_N
    tid_n = tid % BLOCK_N

    # Global row/column indices
    m = m0 + tid_m
    n = n0 + tid_n

    # Validity mask
    mask = (m < M) & (n < N)

    # Initialize accumulator
    acc = tl.zeros((BLOCK_M * BLOCK_N, ), dtype=tl.float32)

    # Number of quantization blocks
    quant_blocks = K // 32

    # Loop over quantization blocks
    for k_block in range(quant_blocks):
        # Base addresses for quantization blocks
        base_A = (m * quant_blocks + k_block) * 34
        base_B = (n * quant_blocks + k_block) * 34

        # Load scaling factors (2 bytes = float16)
        scale_A = load_int8x2_as_fp16(A_ptr + base_A, mask)
        scale_B = load_int8x2_as_fp16(B_ptr + base_B, mask)

        # Compute dot product of 32 int8 weights
        int_dot = tl.zeros((BLOCK_M * BLOCK_N, ), dtype=tl.int32)
        for i in range(32):
            a = tl.load(A_ptr + base_A + 2 + i, mask=mask, other=0)
            b = tl.load(B_ptr + base_B + 2 + i, mask=mask, other=0)
            int_dot += a * b

        # Accumulate contribution
        acc += scale_A.to(tl.float32) * scale_B.to(tl.float32) * int_dot.to(tl.float32)

    # Store results (convert to float16)
    tl.store(C_ptr + m * N + n, acc.to(tl.float16), mask=mask)


def mmq_q8_0(A: torch.Tensor, B: torch.Tensor, M: int, N: int, K: int):
    """
    Triton implementation of q8_0 quantized matrix multiplication: C = A @ B.T
    
    Args:
        A: Quantized matrix A (int8, q8_0 format)
        B: Quantized matrix B (int8, q8_0 format)
        M: Rows of A
        N: Rows of B (columns of B.T)
        K: Columns of A and B (must be divisible by 32)
    
    Returns:
        C: Result matrix (float16, M x N)
    """
    assert K % 32 == 0, "K must be divisible by 32"
    assert A.dtype == torch.int8 and B.dtype == torch.int8
    assert A.numel() == M * K // 32 * 34
    assert B.numel() == N * K // 32 * 34

    # Output matrix
    C = torch.empty((M, N), dtype=torch.float16, device=A.device)

    # Grid dimensions
    grid = (triton.cdiv(M, 16), triton.cdiv(N, 16))

    # Launch kernel
    mmq_q8_0_kernel[grid](A, B, C, M, N, K, BLOCK_M=16, BLOCK_N=16)

    return C
