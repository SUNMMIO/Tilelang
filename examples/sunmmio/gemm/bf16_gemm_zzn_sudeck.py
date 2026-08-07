"""BF16 GEMM with a ZZN B matrix on the Sunmmio SuDeck backend."""

import argparse

import torch
import torch_sunmmio  # noqa: F401 - registers the "sunmmio" device
from torch_sunmmio import sunmmio as sm

import tilelang
import tilelang.language as T

from bf16_gemm_zzn import _validate_config, matmul_persistent_zzn


@tilelang.jit(target="sunmmio", execution_backend="sunmmio")
def bf16_gemm_zzn_sudeck(M, N, K, block_M, block_N, block_K):
    return matmul_persistent_zzn(
        M,
        N,
        K,
        block_M,
        block_N,
        block_K,
        dtype=T.bfloat16,
        accum_dtype=T.float32,
    )


def main(M=256, N=256, K=256, block_M=32, block_N=64, block_K=64):
    _validate_config(M, N, K, block_M, block_N, block_K)

    torch.manual_seed(0)
    a = torch.randn((M, K), dtype=torch.float32).to(torch.bfloat16)
    b = torch.randn((K, N), dtype=torch.float32).to(torch.bfloat16)
    expected = a.float() @ b.float()

    kernel = bf16_gemm_zzn_sudeck(
        M,
        N,
        K,
        block_M=block_M,
        block_N=block_N,
        block_K=block_K,
    )

    placement = sm.placement.full_shard(0, 1)
    with sm.spec(layout=sm.layout.zz(0, 1), placement=placement):
        a_dev = a.to("sunmmio")
        c_dev = sm.empty(M, N, dtype=torch.float32)
    with sm.spec(
        layout=sm.layout.zzn(0, 1, block_K // 32, block_N // 32),
        placement=placement,
    ):
        b_dev = b.to("sunmmio")

    kernel(a_dev, b_dev, c_dev)
    actual = c_dev.cpu()

    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)
    print(f"bf16_gemm_zzn_sudeck PASS: shape=({M}, {N}, {K}), blocks=({block_M}, {block_N}, {block_K})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, default=256)
    parser.add_argument("--n", type=int, default=256)
    parser.add_argument("--k", type=int, default=256)
    parser.add_argument("--block-m", type=int, default=32)
    parser.add_argument("--block-n", type=int, default=64)
    parser.add_argument("--block-k", type=int, default=64)
    args, _ = parser.parse_known_args()
    main(args.m, args.n, args.k, args.block_m, args.block_n, args.block_k)
