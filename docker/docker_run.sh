#!/usr/bin/env bash

set -euo pipefail

# Usage: ./docker_run.sh [OPTIONS]
# Options:
#   -ws WORKSPACE_DIR  Workspace directory (default: ~/workspace)
#   -n CONTAINER_NAME  Container name (default: tilelang-mesh-dev)
#   -i IMAGE_NAME      Image name (default: sunmmio/tilelang-mesh:cuda-dev)
# Example: ./docker_run.sh -ws /path/to/Tilelang -n tilelang-mesh-dev

# Default values
while [[ $# -gt 0 ]]; do
  case "$1" in
    -ws)
      WORKSPACE_DIR="$2"
      shift 2
      ;;
    -n)
      CONTAINER_NAME="$2"
      shift 2
      ;;
    -i)
      IMAGE_NAME="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1" >&2
      exit 2
      ;;
  esac
done

IMAGE_NAME=${IMAGE_NAME:-sunmmio/tilelang-mesh:cuda-dev}
WORKSPACE_DIR=${WORKSPACE_DIR:-${PWD}}
CONTAINER_NAME=${CONTAINER_NAME:-tilelang-mesh-dev}

echo "Starting TileLang-Mesh development container in background..."
echo "  Image: ${IMAGE_NAME}"
echo "  Workspace: ${WORKSPACE_DIR}"

docker run -d \
  --gpus all \
  --ipc=host \
  --name "${CONTAINER_NAME}" \
  --mount "type=bind,src=${WORKSPACE_DIR},dst=/workspace/Tilelang" \
  --workdir /workspace/Tilelang \
  "${IMAGE_NAME}" \
  tail -f /dev/null

echo "Enter with: docker exec -it ${CONTAINER_NAME} bash"
