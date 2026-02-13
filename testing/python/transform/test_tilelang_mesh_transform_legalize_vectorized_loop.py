from tilelang import tvm as tvm
import tilelang as tl
import tilelang.language as T
import tilelang.testing
from tilelang.utils.target import determine_target


def vectorize_access_legalize(M: int = 64, N: int = 1024):
    dtype = "float32"
    vec_len = 512

    @T.prim_func
    def main(A: T.Tensor((M, N, vec_len), dtype="float32"),):
        with T.Kernel(1, 1, threads=M) as (bx, by):
            A_shared = T.alloc_shared((M, N, vec_len), dtype=dtype)
            tid = T.get_thread_binding()
            for j in T.serial(N):
                for v in T.vectorized(vec_len):
                    A_shared[tid, j, v] = A[tid, j, v]

    @T.prim_func
    def expected(A: T.Tensor((M, N, vec_len), dtype="float32"),):
        with T.Kernel(1, 1, threads=M) as (bx, by):
            A_shared = T.alloc_shared((M, N, vec_len), dtype=dtype)
            tid = T.get_thread_binding()
            for j, v_2 in T.grid(M, vec_len // 256):
                for vec in T.vectorized(256):
                    A_shared[tid, j, v_2 * 256 + vec] = A[tid, j, v_2 * 256 + vec]

    return main, expected


def assert_vectorize_access(M: int = 64, N: int = 1024):
    func, expected = vectorize_access_legalize(M, N)
    mod = tvm.IRModule({func.attrs["global_symbol"]: func})
    mod.show()
    target = determine_target("Sunmmio", return_object=True)
    # mod = tvm.tir.transform.BindTarget(target)(mod)
    # transformed = tl.transform.LegalizeVectorizedLoop()(mod)
    with tvm.target.Target(target):
        transformed = tl.transform.LegalizeVectorizedLoop()(mod)
    transformed.show()
    print("Expected:{}".format(expected.body))
    tvm.ir.assert_structural_equal(transformed["main"].body, expected.body)


def test_vectorize_access():
    assert_vectorize_access(1024, 1024)


if __name__ == "__main__":
    tilelang.testing.main()
