import torch
torch.random.manual_seed(0)
M = 1
N = 128
K = 128
GROUP = 4
Bits = 1
input_fp16 = torch.rand(M, K, dtype=torch.float16, device="cuda")
# -1 or 1
weight_int1 = torch.randint(0, 2, (N, K), dtype=torch.int8, device="cuda")
weight_int4 = torch.zeros((N, K // GROUP), dtype=torch.int8, device="cuda")

for n in range(N):
    for k in range(K // GROUP):
        weight_chunk = weight_int1[n, k * GROUP : (k + 1) * GROUP]
        weight_4bit = 0
        for i in range(GROUP):
            weight_4bit |= (weight_chunk[i] << (GROUP-1-i))
        weight_int4[n, k] = weight_4bit

ref_output = torch.matmul(input_fp16, weight_int1.T.to(torch.float16))

print(ref_output)

# create precompute table
table_fp16 = torch.zeros((M, K // GROUP, 2**GROUP), dtype=torch.float16, device="cuda")
for k in range(K // GROUP):
    table_fp16[:, k, 0] = 0
    table_fp16[:, k, 1] = input_fp16[:, k * GROUP + 3]
    table_fp16[:, k, 2] = input_fp16[:, k * GROUP + 2]
    table_fp16[:, k, 3] = input_fp16[:, k * GROUP + 2] + input_fp16[:, k * GROUP + 3]
    table_fp16[:, k, 4] = input_fp16[:, k * GROUP + 1]
    table_fp16[:, k, 5] = input_fp16[:, k * GROUP + 1] + input_fp16[:, k * GROUP + 3]
    table_fp16[:, k, 6] = input_fp16[:, k * GROUP + 1] + input_fp16[:, k * GROUP + 2]
    table_fp16[:, k, 7] = input_fp16[:, k * GROUP + 1] + input_fp16[:, k * GROUP + 2] + input_fp16[:, k * GROUP + 3]
    table_fp16[:, k, 8] = input_fp16[:, k * GROUP + 0] 
    table_fp16[:, k, 9] = input_fp16[:, k * GROUP + 0] + input_fp16[:, k * GROUP + 3]
    table_fp16[:, k, 10] = input_fp16[:, k * GROUP + 0] + input_fp16[:, k * GROUP + 2]
    table_fp16[:, k, 11] = input_fp16[:, k * GROUP + 0] + input_fp16[:, k * GROUP + 2] + input_fp16[:, k * GROUP + 3]
    table_fp16[:, k, 12] = input_fp16[:, k * GROUP + 0] + input_fp16[:, k * GROUP + 1]
    table_fp16[:, k, 13] = input_fp16[:, k * GROUP + 0] + input_fp16[:, k * GROUP + 1] + input_fp16[:, k * GROUP + 3]
    table_fp16[:, k, 14] = input_fp16[:, k * GROUP + 0] + input_fp16[:, k * GROUP + 1] + input_fp16[:, k * GROUP + 2]
    table_fp16[:, k, 15] = input_fp16[:, k * GROUP + 0] + input_fp16[:, k * GROUP + 1] + input_fp16[:, k * GROUP + 2] + input_fp16[:, k * GROUP + 3]

from bitblas import tvm as tvm
from tvm import tl
import tvm.tl.language as T
TABLE_shape = (M, K // GROUP, 2**GROUP)
dtype_table = "float16"
B_shape = (N, K // GROUP)
dtype_b = "int8"
threads = 32

N_Tile = 128
K_Tile = 16


@T.prim_func
def main_nTile_kTile_threads(TABLE: T.Buffer(TABLE_shape, dtype_table), B: T.Buffer(B_shape, dtype_b), C: T.Buffer((M, N), dtype_table)):
    accum_res = T.alloc_fragment((N_Tile // threads,), dtype_table, "local")
    query = T.alloc_fragment((1,), "int8", "local")
    with T.Kernel(M, T.ceildiv(N, N_Tile), threads=threads) as (bx, by):
        for n in T.serial(N_Tile // threads):
            accum_res[n] = 0
        for ko in T.serial(((K // GROUP) // K_Tile)):
            for n in T.serial(N_Tile // threads):
                for tx in T.thread_binding(0, threads, thread="threadIdx.x"):
                    for ki in T.serial(K_Tile):
                        query[0] = B[by * N_Tile + (n * threads + tx), ko * K_Tile + ki]
                        accum_res[n] += TABLE[
                            bx, ko * K_Tile + ki, query[0]
                        ]
        for n in T.serial(N_Tile // threads):
            for t in T.thread_binding(0, threads, thread="threadIdx.x"):
                C[bx, by * N_Tile + (n * threads + t)] = accum_res[n]


@tvm.register_func(func_name="tvm_callback_cuda_postproc", override=True)
def tvm_callback_cuda_postproc(code, _):
    print(code)
    return code


mod, params = tl.lower(main_nTile_kTile_threads)
mod = tl.Profiler(mod, params, [2], tl.TensorSupplyType.Integer)

with open("debug/kernel.cu", "w") as f:
    f.write(mod.mod.imported_modules[0].get_source())

# out = mod.run_once()
cuda_table = table_fp16.cuda()
cuda_weight = weight_int4.cuda()

cuda_output = mod.func(cuda_table, cuda_weight)
print("cuda_output:", cuda_output)

# assert close

torch.testing.assert_close(ref_output, cuda_output, rtol=1e-2, atol=1e-2)
