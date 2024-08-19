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
threads = 256
N_Tile_factor = 1
reduce_k = 8

query_vectorize_size = 8 # as we should fetch int8
K_Tile = query_vectorize_size * 2

thread_num_y = reduce_k
thread_num_x = threads // thread_num_y
N_Tile = N_Tile_factor * thread_num_x

print(source)

@T.prim_func
def main_nTile_kTile_threads_reducek(TABLE: T.Buffer(TABLE_shape, dtype_table), B: T.Buffer(B_shape, dtype_b), C: T.Buffer((M, N), dtype_table)):
    accum_res = T.alloc_fragment(
        (N_Tile // thread_num_x,), dtype_table, "local"
    )
    reduced_accum_res = T.alloc_fragment(
    0, dtype_table, "local"
    )
    packed_query = T.alloc_fragment((query_vectorize_size,), "int8", "local")
    query = T.alloc_fragment((query_vectorize_size * 2,), "int8", "local")
    with T.Kernel(M, T.ceildiv(N, N_Tile), threads=threads) as (bx, by):
        for n in T.serial(N_Tile // thread_num_x):
            accum_res[n] = T.float16(0)
        for kr in T.thread_binding(0, reduce_k, thread="threadIdx.y", annotations={"pragma_import_c": source}):
            for ko in T.serial((((K // GROUP) // reduce_k) // K_Tile)):
                for tx in T.thread_binding(0, thread_num_x, thread="threadIdx.x"):
                    for n in T.serial(N_Tile // thread_num_x):
                        for v in T.vectorized(query_vectorize_size):
                            packed_query[v] = B[
                                by * N_Tile + (n * thread_num_x + tx), (ko * reduce_k + kr) * (K_Tile // 2) + v
                            ]
                        T.call_extern("handle", "decode_i4u_to_i8s", T.address_of(packed_query[0]), T.address_of(query[0]), 16)
                        for v in T.serial(query_vectorize_size * 2):
                            accum_res[n] += TABLE[bx, ko * reduce_k * K_Tile + kr * K_Tile + v, query[v]]

            for n in T.serial(N_Tile // thread_num_x):
                T.attr(
                    T.comm_reducer(lambda x, y: x + y, [T.float16(0)]),
                    "reduce_scope",
                    T.reinterpret(T.uint64(0), dtype="handle"),
                )
                T.evaluate(
                    T.tvm_thread_allreduce(
                        T.uint32(1),
                        accum_res[n],
                        True,
                        reduced_accum_res[0],
                        kr,
                        dtype="handle",
                    )
                )
                if kr == 0:
                    accum_res[n] = reduced_accum_res[0]
            if kr == 0:
                for n in T.serial(N_Tile // thread_num_x):
                    for t in T.thread_binding(0, thread_num_x, thread="threadIdx.x"):
                        C[bx, by * N_Tile + (n * thread_num_x + t)] = accum_res[n]

@tvm.register_func(func_name="tvm_callback_cuda_postproc", override=True)
def tvm_callback_cuda_postproc(code, _):
    print(code)
    return code


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