from copy import deepcopy

from tvm import tir

import tilelang.language as T


def mesh_tensor_functions():
    _mesh_tensor_info = {}

    def annotate_mesh_tensor_info(mesh_tensor_info: dict):
        nonlocal _mesh_tensor_info
        _mesh_tensor_info = {}
        for buffer, info in mesh_tensor_info.items():
            if (
                not isinstance(info, dict)
                or 'block_shape' not in info
                or 'program_id' not in info
                or 'sharding' not in info
            ):
                raise ValueError(f'Invalid mesh tensor info: {info}')
            else:
                _mesh_tensor_info[buffer.data] = deepcopy(info)

        return T.func_attr({'mesh_tensor_info': _mesh_tensor_info})

    def mesh_tensor_copy(
        src: tir.Buffer,
        dst: tir.Buffer,
        *,
        src_coord: tuple[int] | None = None,
        dst_coord: tuple[int] | None = None,
    ):
        nonlocal _mesh_tensor_info
        if src_coord is not None:
            try:
                info = _mesh_tensor_info[src.data]
                block_shape = info['block_shape']
                src = src[tuple(i * b for i, b in zip(src_coord, block_shape))]
            except KeyError as e:
                raise ValueError(
                    f'MeshTensor information for buffer {src} not found.'
                ) from e
        if dst_coord is not None:
            try:
                info = _mesh_tensor_info[dst.data]
                block_shape = info['block_shape']
                dst = dst[tuple(i * b for i, b in zip(dst_coord, block_shape))]
            except KeyError as e:
                raise ValueError(
                    f'MeshTensor information for buffer {dst} not found.'
                ) from e
        return T.copy(src, dst)

    return annotate_mesh_tensor_info, mesh_tensor_copy
