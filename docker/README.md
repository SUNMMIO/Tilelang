# TileLang-Mesh Containers

The versioned CUDA Dockerfiles clone the canonical SUNMMIO repository, initialize only public
submodules, and build with `USE_SUNMMIO=OFF`. They do not require access to NPU-IR. Set
`TILELANG_REF` to the release tag or exact commit that the image must contain.

```bash
git clone https://github.com/SUNMMIO/Tilelang.git
cd Tilelang
docker build \
  --build-arg TILELANG_REF=v0.1.0 \
  -t tilelang-mesh:0.1.0-cu124 \
  -f docker/Dockerfile.cu124 docker
```

Run the NVIDIA image:

```bash
docker run --rm -it \
  --gpus all \
  --ipc=host \
  --shm-size=4G \
  tilelang-mesh:0.1.0-cu124 bash
```

The ROCm Dockerfile uses the repository root as its build context:

```bash
docker build -t tilelang-mesh:rocm -f docker/Dockerfile.rocm .
docker run --rm -it \
  --device=/dev/kfd \
  --device=/dev/dri \
  --ipc=host \
  tilelang-mesh:rocm
```

## Development Image

`Dockerfile.cu130.dev` creates a CUDA development environment from the local repository context. It
does not install TileLang-Mesh automatically, so mount or copy an authorized checkout and select the
backend explicitly:

```bash
docker build -t sunmmio/tilelang-mesh:cuda-dev -f docker/Dockerfile.cu130.dev .

docker run -it --rm \
  --gpus all \
  --ipc=host \
  -v "${PWD}:/workspace/Tilelang" \
  sunmmio/tilelang-mesh:cuda-dev bash
```

Inside an authorized checkout with NPU-IR access:

```bash
cd /workspace/Tilelang
CMAKE_ARGS="-DUSE_CUDA=ON -DUSE_SUNMMIO=ON" python -m pip install -e . -v
```

For a public build, initialize only public submodules and use `USE_SUNMMIO=OFF` as described in the
main [installation guide](../docs/get_started/Installation.md).

The optional `docker/docker_run.sh` helper starts the development image in the background without
opening an SSH service. Use `docker exec -it tilelang-mesh-dev bash` to enter it.
