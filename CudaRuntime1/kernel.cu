
#include "cuda_runtime.h"
#include <cuda_runtime_api.h>
#include "device_launch_parameters.h"

#include <stdio.h>

__global__ void helloKernel()
{
    printf("Hello world\n");
}

__global__ void infoKernel()
{
    printf("I am from %d block, %d thread\n", blockIdx.x, threadIdx.x);
}

__global__ void addKernel(int* arr, int n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        arr[tid] = arr[tid] + tid;
    }
}

int main()
{
    // task 1
    {
        helloKernel << <32, 32 >> > ();
    }
    
    //task 2
    {
        infoKernel << <32, 32>> > ();
    }

    // task 3
    {
        const int size = 1e8;
        int* arr = new int[size];
        int* arr_gpu;
        cudaMalloc((void**)(&arr_gpu), size * sizeof(int));
        cudaMemcpy(arr_gpu, arr, size * sizeof(int), cudaMemcpyHostToDevice);
        const int blockSize = 32;
        const int blockNum = (size + blockSize - 1) / blockSize;
        addKernel <<<blockSize, blockNum >>> (arr_gpu, size);
        cudaMemcpy(arr, arr_gpu, size * sizeof(int), cudaMemcpyDeviceToHost);
        for (int i = 0; i < size; i++) {
            //printf("%d ", arr[i]);
        }
        delete[] arr;
    }

    return 0;
}
