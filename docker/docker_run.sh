#!/usr/bin/bash

# Usage: ./docker_run.sh [OPTIONS]
# Options:
#   -p SSH_PORT        SSH port (default: 2222)
#   -ws WORKSPACE_DIR  Workspace directory (default: ~/workspace)
#   -n CONTAINER_NAME  Container name (default: ccl-dev)
# Example: ./docker_run.sh -p 2222 -ws /path/to/workspace -n jiaqi_tilelang_dev 

# Default values
while [[ $# -gt 0 ]]; do
  case "$1" in
    -p)
      SSH_PORT="$2"
      shift 2
      ;;
    -ws)
      WORKSPACE_DIR="$2"
      shift 2
      ;;
    -n)
      CONTAINER_NAME="$2"
      shift 2
      ;;
    *)
      shift
      ;;
  esac
done

IMAGE_NAME="sunlune/tilelang:cuda"
DOCKER_RT="--runtime nvidia --gpus all"

SSH_PORT=${SSH_PORT:-2222}
WORKSPACE_DIR=${WORKSPACE_DIR:-~/workspace}
CONTAINER_NAME=${CONTAINER_NAME:-ccl-dev}

echo "Starting CCL Development Docker container in background..."
echo "  SSH Port: ${SSH_PORT}"
echo "  Image: ${IMAGE_NAME}"
echo "  Workspace: ${WORKSPACE_DIR}"

# Build docker run command
DOCKER_CMD="docker run -d ${DOCKER_RT} \
    --ipc=host \
    --name ${CONTAINER_NAME} \
    -p ${SSH_PORT}:22 \
    -v \"${WORKSPACE_DIR}:/workspace\" \
    --cap-add=SYS_PTRACE \
    -w /workspace"

# Complete the command
DOCKER_CMD="${DOCKER_CMD} ${IMAGE_NAME} /bin/bash -c \"service ssh start && tail -f /dev/null\""

echo $DOCKER_CMD

# Execute the command
eval $DOCKER_CMD