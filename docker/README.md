# Production Usage

To ease the process of installing all the dependencies, we provide a Dockerfile and a simple guideline to build a Docker image with all of above installed. The Docker image is built on top of Ubuntu 20.04, and it contains all the dependencies required to run the experiments. We only provide the Dockerfile for NVIDIA GPU, and the Dockerfile for AMD GPU will be provided upon request.

```bash
git clone --recursive https://github.com/tile-ai/tilelang TileLang
cd TileLang/docker
# build the image, this may take a while (around 10+ minutes on our test machine)
# replace the version number cu124 with the one you want to use
# replace .cu** with .rocm for AMD GPU
docker build -t tilelang_workspace -f Dockerfile.cu124 .
# run the container
# if it's nvidia
docker run -it --cap-add=SYS_ADMIN --network=host --gpus all --cap-add=SYS_PTRACE --shm-size=4G --security-opt seccomp=unconfined --security-opt apparmor=unconfined --name tilelang_test tilelang_workspace bash
# if it's amd
docker run -it --cap-add=SYS_ADMIN --network=host --device=/dev/kfd --device=/dev/dri  --cap-add=SYS_PTRACE --shm-size=4G --security-opt seccomp=unconfined --security-opt apparmor=unconfined --name tilelang_test tilelang_workspace bash
```

# Development

## Docker build

```shell
cd ..
docker build -t sunlune/tilelang:cuda -f ./docker/Dockerfile.cu130.dev .
```

## Launch Docker Container as a Service

```bash
./docker/docker_run.sh -p 2222 -ws $(pwd) -n jiaqi_tilelang_dev
```

### Enter the container for development

```bash
docker ps
# The terminal output should be as follows
CONTAINER ID   IMAGE                                        COMMAND                  CREATED          STATUS          PORTS                                     NAMES
285477349f8d   sunlune/tilelang:cuda                             "/opt/nvidia/nvidia_…"   17 minutes ago   Up 17 minutes   0.0.0.0:2222->22/tcp, [::]:2222->22/tcp   jiaqi_tilelang_dev

docker exec -it ${your_container_id} bash # in this case your_container_id = 285477349f8d
```

### Connect container via ssh

As we use jump host to connect to the remote machine, the ssh port is not accessible directly. We need to config the local machine's ~/.ssh/config, add the following per your need:

```yaml
Host bj3080_jumphost
  HostName 192.168.3.214
  Port 11023
  User <user>

# Docker 容器配置
Host bj3080_docker_dev
  HostName localhost
  Port <port>
  User root
  ProxyJump bj3080_jumphost
```

Then you can connect with
```bash
ssh bj3080_docker_dev

# or use vscode to connect the docker directly
```
You can also use one line to connect, as below
```bash
ssh -J <user>@192.168.3.214:11023 root@localhost -p <port>
```

### Launch interactive docker container

```shell
docker run -it --runtime nvidia --gpus all \
  -v "$(dirname "$PWD"):/workspace" \
  --name jiaqi_tilelang_dev \
  --ipc=host \
  sunlune/tilelang:cuda \
  bash
```