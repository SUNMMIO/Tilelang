import pytest
from tilelang.layout.hierarchical_layout import HierarchicalLayout


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
