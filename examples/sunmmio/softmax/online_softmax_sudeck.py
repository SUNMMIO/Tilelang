"""Online softmax on the Sunmmio SuDeck (real-device) backend, driven purely by torch
and torch_sunmmio.

Same kernel as ``online_softmax.py`` (which targets the sunsim simulator); here we compile
with ``execution_backend="sunmmio"`` and run on ``torch_sunmmio`` device tensors. The kernel
launches on the current torch-sunmmio stream.
"""

import argparse

import torch
import torch_sunmmio  # noqa: F401  registers the "sunmmio" device
from torch_sunmmio import sunmmio as sm

import tilelang
import tilelang.language as T

from online_softmax import softmax_kernel


@tilelang.jit(target="sunmmio", execution_backend="sunmmio")
def online_softmax_sudeck(M, N, block_M, block_N, dtype):
    return softmax_kernel(M, N, block_M, block_N, dtype)


def main(M, N) -> None:
    torch.manual_seed(0)
    x = (torch.rand(M, N) * 8.0 - 4.0).to(torch.bfloat16)

    kernel = online_softmax_sudeck(M, N, block_M=256, block_N=256, dtype=T.bfloat16)

    # Match the kernel's MeshTensor(placement=MeshShardingPolicy(y=0, x=1), layout=zz):
    # zz block layout + full 2D shard across the mesh.
    with sm.spec(layout=sm.layout.zz(0, 1), placement=sm.placement.full_shard(0, 1)):
        x_dev = x.to("sunmmio")
        y_dev = sm.empty(M, N, dtype=torch.bfloat16)

    kernel(x_dev, y_dev)
    y = y_dev.cpu().float()

    ref = torch.softmax(x.float(), dim=1)
    torch.testing.assert_close(y, ref, rtol=5e-2, atol=5e-3)
    print(f"online_softmax_sudeck PASS: shape=({M}, {N}) matched torch.softmax within tolerance")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, default=1024)
    parser.add_argument("--n", type=int, default=1024)
    args, _ = parser.parse_known_args()
    main(args.m, args.n)
