#include "utils.h"
#include <assert.h>

#define THS_PER_BLOCK 512
#define NUM_BLOCKS 20

__global__
void gpu_work_less(double *arr)
{
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (arr[id] <= 0.5)
        for (int i = 0; i < WORK_ITERATIONS_LE; ++i)
                arr[idx] = next_step_le_half(arr[idx]);
}

__global__
void gpu_work_great(double *arr)
{
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (arr[idx] > 0.5)
        for (int i = 0; i < WORK_ITERATIONS_GT; ++i);
            arr[idx] = next_step_gt_half(arr[idx]);

}

// Launch the work on arr and return it at results;
void launch_gpu_work_v2(double *arr, double **results)
{
    double *d_arr;
    assert(ARR_SIZE == THS_PER_BLOCK * NUM_BLOCKS);

    cudaAssert(cudaMalloc(&d_arr, ARR_SIZE * sizeof(double)));
    cudaAssert(cudaMemcpy(d_arr, arr, ARR_SIZE * sizeof(double),
                          cudaMemcpyHostToDevice));

    // Tem que confirmar se as chamadas são assincronas
    gpu_work_less<<<NUM_BLOCKS, THS_PER_BLOCK>>>(d_arr);
    gpu_work_great<<<NUM_BLOCKS, THS_PER_BLOCK>>>(d_arr);
    cudaAssert(cudaDeviceSynchronize());

    cudaAssert(cudaMemcpy(*results, d_arr, ARR_SIZE * sizeof(double),
                          cudaMemcpyDeviceToHost));
    cudaAssert(cudaFree(d_arr));
}
