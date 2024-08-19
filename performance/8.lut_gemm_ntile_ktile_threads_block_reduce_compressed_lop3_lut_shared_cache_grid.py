import torch
import itertools
import tqdm

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

search_space = {
    "threads": [32, 64, 128, 256],
    "N_Tile_factor": [1, 2, 3, 4, 5, 6, 7, 8, 16],
    "reduce_k": [2, 4, 8, 16],
}

keys = search_space.keys()
values = search_space.values()
combinations = list(itertools.product(*values))

combinations_dicts = [dict(zip(keys, combination)) for combination in combinations]

# for combination in combinations_dicts:
#     print(combination)
print(len(combinations_dicts))

print(source)
tuning_results = {}

min_time = 1e9
min_combination = None
sucess_combinations = []


# out = mod.run_once()
cuda_table = table_fp16.cuda()
cuda_weight = weight_int4_interleaved.cuda()

# set up tqdm
pbar = tqdm.tqdm(combinations_dicts)
for combination in pbar:
    threads = combination["threads"]
    N_Tile_factor = combination["N_Tile_factor"]
    reduce_k = combination["reduce_k"]
    
    query_vectorize_size = 8 # as we should fetch int8
    K_Tile = query_vectorize_size * 2

    thread_num_y = reduce_k
    thread_num_x = threads // thread_num_y
    N_Tile = N_Tile_factor * thread_num_x
    try:
        assert (((K // GROUP) // reduce_k) // K_Tile) > 0, "assertion failed, please adjust the parameters"
        assert (N_Tile // thread_num_x) > 0, "assertion failed, please adjust the parameters"
        
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
            cached_table = T.alloc_fragment((reduce_k * K_Tile, 2**GROUP), dtype_table, "shared")
            with T.Kernel(M, T.ceildiv(N, N_Tile), threads=threads) as (bx, by):
                tx = T.launch_thread("threadIdx.x", thread_num_x)
                for n in T.serial(N_Tile // thread_num_x):
                    accum_res[n] = T.float16(0)
                for kr in T.thread_binding(0, reduce_k, thread="threadIdx.y", annotations={"pragma_import_c": source}):
                    for ko in T.serial((((K // GROUP) // reduce_k) // K_Tile)):
                        for v0 in T.thread_binding(0, reduce_k, thread="threadIdx.y"):
                            for v1 in T.thread_binding(0, thread_num_x, thread="threadIdx.x"):
                                for v2 in T.vectorized(8):
                                    o_v2 = (v1 % 2) * 8 + v2
                                    o_v1 = (v1 // 2)
                                    o_v0 = v0
                                    cached_table[o_v0 * K_Tile + o_v1, o_v2] = TABLE[bx, (ko * reduce_k + o_v0) * K_Tile + o_v1, o_v2]
                        T.tvm_storage_sync("shared")
                        for n in T.serial(N_Tile // thread_num_x):
                            for v in T.vectorized(query_vectorize_size):
                                packed_query[v] = B[
                                    by * N_Tile + (n * thread_num_x + tx), (ko * reduce_k + kr) * (K_Tile // 2) + v
                                ]
                            T.call_extern("handle", "decode_i4u_to_i8s", T.address_of(packed_query[0]), T.address_of(query[0]), 16)
                            for v in T.serial(query_vectorize_size * 2):
                                accum_res[n] += cached_table[kr * K_Tile + v, query[v]]

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


        mod, params = tl.lower(main_nTile_kTile_threads_reducek)
        mod = tl.Profiler(mod, params, [2], tl.TensorSupplyType.Integer)

        latency = mod.do_bench(mod.func)

        print(f"For combination {combination}, time is {latency} ms")
        tuning_results["-".join([str(v) for v in combination.values()])] = latency
        if latency < min_time:
            min_time = latency
            min_combination = combination
        sucess_combinations.append(combination)
    except Exception as e:
        del e
        print(f"Failed for combination {combination}")
        continue

print(f"Minimum time is {min_time} for combination {min_combination}")
