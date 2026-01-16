import pytest

from tilelang import tvm as tvm
from tilelang.layout import HierarchicalLayout, make_hierarchical_layout


@pytest.mark.parametrize(
    "hdims, hstrides, groups, expected_logical_shape, expected_str",
    [
        (
            [8, 128, 8, 128],
            [1024, 1, 8192, 128],
            [(0, 2), (2, 4)],
            (1024, 1024),
            "HierarchicalLayout(hdims=((8, 128), (8, 128)), hstrides=((1024, 1), (8192, 128)))",
        ),
        (
            [4, 8, 16, 32],
            [1, 4, 32, 512],
            [(0, 1), (1, 3), (3, 4)],
            (4, 128, 32),
            "HierarchicalLayout(hdims=((4), (8, 16), (32)), hstrides=((1), (4, 32), (512)))",
        ),
        ([2, 4, 6], [24, 6, 1], [(0, 3)],
         (48,), "HierarchicalLayout(hdims=((2, 4, 6)), hstrides=((24, 6, 1)))"),
        ([2, 4, 6], [24, 6, 1], [(0, 3)],
         (48,), "HierarchicalLayout(hdims=((2, 4, 6)), hstrides=((24, 6, 1)))"),
        ([2, 2, 2, 2], [8, 2, 4, 1], [(0, 2), (2, 4)], (
            4,
            4,
        ), "HierarchicalLayout(hdims=((2, 2), (2, 2)), hstrides=((8, 2), (4, 1)))"),
        ([], [], [], (), "HierarchicalLayout(hdims=(), hstrides=())"),
        (
            [4, 1, 1, 5],
            [5, 20, 20, 1],
            [(0, 1), (1, 3), (3, 4)],
            (4, 1, 5),
            "HierarchicalLayout(hdims=((4), (1, 1), (5)), hstrides=((5), (20, 20), (1)))",
        ),
    ],
)
def test_hierarchical_layout_properties(hdims, hstrides, groups, expected_logical_shape,
                                        expected_str):
    layout = HierarchicalLayout(hdims, hstrides, groups)
    assert layout.logical_shape == expected_logical_shape
    assert str(layout) == expected_str


def test_hierarchical_layout_invalid_init():
    with pytest.raises(AssertionError, match="hdims and hstrides must have the same length"):
        HierarchicalLayout(hdims=[1, 2], hstrides=[1], groups=[(0, 2)])


def test_make_hierarchical_layout_invalid_arg():
    with pytest.raises(ValueError, match="Invalid argument type for make_hierarchical_layout"):
        make_hierarchical_layout(123)


@pytest.mark.parametrize(
    "hdims, hstrides, groups, logical_indices, expected_hierarchical_indices",
    [
        (
            [8, 128, 8, 128],
            [1024, 1, 8192, 128],
            [(0, 2), (2, 4)],
            [1, 1],
            [0, 1, 0, 1],
        ),
        (
            [8, 128, 8, 128],
            [1024, 1, 8192, 128],
            [(0, 2), (2, 4)],
            [129, 257],  # 1*128+1, 2*128+1
            [1, 1, 2, 1],
        ),
        (
            [4, 8, 16, 32],
            [1, 4, 32, 512],
            [(0, 1), (1, 3), (3, 4)],
            [3, 20, 10],  # 20 = 1*16+4
            [3, 1, 4, 10],
        ),
        (
            [2, 4, 6],
            [24, 6, 1],
            [(0, 3)],
            [29],  # 29 = 1*24+1*6-1 -> no. 29 = 1*4*6 + 0*6 + 5 -> [1,0,5]. decompose is weird.
            # 29 -> rem=29. f=6: 29%6=5, rem=4. f=4: 4%4=0, rem=1. f=2: 1%2=1, rem=0 -> [1,0,5]
            [1, 0, 5],
        ),
        ([2, 2, 2, 2], [8, 2, 4, 1], [(0, 2), (2, 4)], [
            2,
            1,
        ], [1, 0, 0, 1]),
        ([2, 2, 2, 2], [8, 2, 4, 1], [(0, 2), (2, 4)], [
            2,
            2,
        ], [1, 0, 1, 0]),
        ([2, 2, 2, 2], [8, 2, 4, 1], [(0, 2), (2, 4)], [
            0,
            2,
        ], [0, 0, 1, 0]),
        ([2, 2, 2, 2], [8, 2, 4, 1], [(0, 2), (2, 4)], [3, 1], [1, 1, 0, 1]),
        ([2, 2, 2, 2], [8, 2, 4, 1], [(0, 2), (2, 4)], [3, 1], [1, 1, 0, 1]),
        (
            [4, 1, 1, 5],
            [5, 20, 20, 1],
            [(0, 1), (1, 3), (3, 4)],
            [3, 0, 2],
            [3, 0, 0, 2],
        ),
        (
            [2, 2],
            [1, 0],
            [(0, 2)],
            [3],
            [1, 1],
        )
    ],
)
def test_hierarchical_layout_index_conversion(hdims, hstrides, groups, logical_indices,
                                              expected_hierarchical_indices):
    layout = HierarchicalLayout(hdims, hstrides, groups)

    # Test logical to hierarchical conversion
    hierarchical_indices = layout.get_hierarchical_indices(logical_indices)
    assert hierarchical_indices == expected_hierarchical_indices

    # Test hierarchical to logical conversion (inverse)
    reconstructed_logical_indices = layout.get_logical_indices(hierarchical_indices)
    assert reconstructed_logical_indices == logical_indices


@pytest.mark.parametrize("hdims, hstrides, groups, hierarchical_indices, expected_logical_indices",
                         [
                             (
                                 [8, 128, 8, 128],
                                 [1024, 1, 8192, 128],
                                 [(0, 2), (2, 4)],
                                 [1, 1, 2, 1],
                                 [129, 257],
                             ),
                             (
                                 [2, 2, 2, 2],
                                 [8, 2, 4, 1],
                                 [(0, 2), (2, 4)],
                                 [1, 0, 0, 1],
                                 [
                                     2,
                                     1,
                                 ],
                             ),
                         ])
def test_hierarchical_layout_inverse_index_conversion(hdims, hstrides, groups, hierarchical_indices,
                                                      expected_logical_indices):
    layout = HierarchicalLayout(hdims, hstrides, groups)

    # Test hierarchical to logical conversion
    logical_indices = layout.get_logical_indices(hierarchical_indices)
    assert logical_indices == expected_logical_indices

    # Test logical to hierarchical conversion (inverse)
    reconstructed_hierarchical_indices = layout.get_hierarchical_indices(logical_indices)
    assert reconstructed_hierarchical_indices == hierarchical_indices


@pytest.mark.parametrize(
    "hdims, hstrides, groups, logical_indices",
    [
        (
            [2, 2, 2, 2],
            [8, 2, 4, 1],
            [(0, 2), (2, 4)],
            [2, 1],
        ),
        (
            [8, 128, 8, 128],
            [1024, 1, 8192, 128],
            [(0, 2), (2, 4)],
            [129, 257],
        ),
        (
            [4, 8, 16, 32],
            [1, 4, 32, 512],
            [(0, 1), (1, 3), (3, 4)],
            [3, 20, 10],
        ),
        (
            [2, 4, 6],
            [24, 6, 1],
            [(0, 3)],
            [29],
        ),
        (
            [4, 1, 1, 5],
            [5, 20, 20, 1],
            [(0, 1), (1, 3), (3, 4)],
            [3, 0, 2],
        ),
        (
            [2, 2],
            [1, 0],
            [(0, 2)],
            [3],
        ),
    ],
)
def test_hierarchical_layout_cpp_apply(hdims, hstrides, groups, logical_indices):
    py_layout = HierarchicalLayout(hdims, hstrides, groups)
    cpp_layout = make_hierarchical_layout(py_layout)

    h_indices = py_layout.get_hierarchical_indices(logical_indices)
    expected_offset = sum(h * s for h, s in zip(h_indices, py_layout.hstrides))

    # Convert logical_indices to a list of PrimExpr (IntImm)
    logical_indices_expr = [tvm.tir.IntImm("int32", i) for i in logical_indices]

    # Use map_forward_index instead of apply
    offset_expr = cpp_layout.map_forward_index(logical_indices_expr)[0]

    # The result of map_forward_index is a PrimExpr, we need to simplify and get the value
    analyzer = tvm.arith.Analyzer()
    simplified_offset = analyzer.simplify(offset_expr)
    assert simplified_offset.value == expected_offset
