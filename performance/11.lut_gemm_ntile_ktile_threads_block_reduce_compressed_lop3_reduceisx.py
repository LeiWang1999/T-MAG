import torch

torch.random.manual_seed(0)
M = 1
N = 16384
K = 16384
GROUP = 4
Bits = 1
table_fp16 = torch.randn((M, K // GROUP, 2**GROUP), dtype=torch.float16, device="cuda")
weight_int4_interleaved = torch.zeros((N, (K // GROUP) // 2), dtype=torch.int8, device="cuda")

from bitblas import tvm as tvm
from tvm import tl
import tvm.tl.language as T
from bitblas.gpu.intrin.lop3 import get_lop3_intrin_group

lop3_info = get_lop3_intrin_group(
    out_dtype="int8",
    source_format="uint",
    source_bit=4,
    storage_dtype="int8",
)

source = lop3_info["c_source"]


TABLE_shape = (M, K // GROUP, 2**GROUP)
dtype_table = "float16"
B_shape = (N, (K // GROUP) // 2)
dtype_b = "int8"
num_warps = 2
warp_size = 32
threads = num_warps * warp_size

query_vectorize_size = 8 # as we should fetch int8
K_Tile = query_vectorize_size * 2

thread_num_x = warp_size
N_Chunk = 1
N_Tile = num_warps * N_Chunk


print("N_Tile:", N_Tile, "K_Tile:", K_Tile, "thread_num_x:", thread_num_x)
assert (((K // GROUP) // thread_num_x) // K_Tile) > 0, "K_Tile is too large"

@T.prim_func
def main_nTile_kTile_threads_reducek(TABLE: T.Buffer(TABLE_shape, dtype_table), B: T.Buffer(B_shape, dtype_b), C: T.Buffer((M, N), dtype_table)):
    accum_res = T.alloc_fragment(
        (N_Tile // num_warps), dtype_table, "local"
    )
    reduced_accum_res = T.alloc_fragment(
       0, dtype_table, "local"
    )
    packed_query = T.alloc_fragment((query_vectorize_size,), "int8", "local")
    query = T.alloc_fragment((query_vectorize_size * 2,), "int8", "local")
    bx = T.thread_binding(0, M, thread="blockIdx.x")
    by = T.thread_binding(0, T.ceildiv(N, N_Tile), thread="blockIdx.y")
    for n in T.serial(N_Tile // num_warps):
        accum_res[n] = T.float16(0)
    for no in T.serial(N_Tile // num_warps):
        for ni in T.thread_binding(0, num_warps, thread="threadIdx.y"):
            # for kr in T.serial(warp_size, annotations={"pragma_import_c": source}):
            for kr in T.thread_binding(0, warp_size, thread="threadIdx.x", annotations={"pragma_import_c": source}):
                for ko in T.serial((((K // GROUP) // warp_size) // K_Tile)):
                    for v in T.vectorized(query_vectorize_size):
                        packed_query[v] = B[
                            by * N_Tile + no * num_warps + ni, (ko * warp_size + kr) * query_vectorize_size + v
                        ]
                    T.call_extern("handle", "decode_i4u_to_i8s", T.address_of(packed_query[0]), T.address_of(query[0]), 16)
                    for v in T.serial(query_vectorize_size * 2):
                        accum_res[no] += TABLE[bx, (ko * warp_size + kr) * K_Tile + v, query[v]]
                T.attr(
                    T.comm_reducer(lambda x, y: x + y, [T.float16(0)]),
                    "reduce_scope",
                    T.reinterpret(T.uint64(0), dtype="handle"),
                )
                T.evaluate(
                    T.tvm_thread_allreduce(
                        T.uint32(1),
                        accum_res[no],
                        True,
                        reduced_accum_res[0],
                        kr,
                        dtype="handle",
                    )
                )
                if kr == 0:
                    accum_res[no] = reduced_accum_res[0]
                if kr == 0:
                    C[bx, by * N_Tile + no * num_warps + ni] = accum_res[no]



@tvm.register_func(func_name="tvm_callback_cuda_postproc", override=True)
def tvm_callback_cuda_postproc(code, _):
    print(code)
    # code = code.replace("TABLE[((((ko * 8192) + (((int)threadIdx.x) * 256)) + (v * 16)) + ((int)query[v]))]);", "TABLE[((((ko * 8192) + (v * 512)) + ((int)threadIdx.x * 16)) + ((int)query[v]))]);")
    return code

print(main_nTile_kTile_threads_reducek)
mod, params = tl.lower(main_nTile_kTile_threads_reducek)
mod = tl.Profiler(mod, params, [2], tl.TensorSupplyType.Integer)

with open("debug/kernel.cu", "w") as f:
    f.write(mod.mod.imported_modules[0].get_source())

# out = mod.run_once()
cuda_table = table_fp16.cuda()
cuda_weight = weight_int4_interleaved.cuda()

cuda_output = mod.func(cuda_table, cuda_weight)
print("cuda_output:", cuda_output)

latency = mod.do_bench(mod.func)

print("latency: ", latency)