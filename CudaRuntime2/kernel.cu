
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <omp.h>
#include <vector>

#include <stdio.h>

__global__ void saxpyKernelGpu(int n, float alpha, float* x, int incx, float* y, int incy)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int ix = tid * incx;
    int iy = tid * incy;
    if (ix < n && iy < n) {
        y[iy] += x[ix] * alpha;
    }
}

__global__ void daxpyKernelGpu(int n, double alpha, double* x, int incx, double* y, int incy)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int ix = tid * incx;
    int iy = tid * incy;
    if (ix < n && iy < n) {
        y[iy] += x[ix] * alpha;
    }
}

void daxpyRun();
void saxpyRun();

int main()
{
    /*cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    printf("%d\n", deviceProp.warpSize);*/
    printf("Daxpy:\n");
    daxpyRun();
    printf("------------------\n");
    printf("Saxpy:\n");
    saxpyRun();
}

void daxpyRun() {
    std::vector<int> sizes = { (int)1e6, (int)1e7, (int)1e8, (int)2e8 };
    for (auto size : sizes)
    {
        printf("size: %d\n", size);
        for (int blockSize = 8; blockSize <= 256; blockSize *= 2)
        {
            double* arr = new double[size];
            for (int i = 0; i < size; i++) {
                arr[i] = i;
            }
            double* gpu_arr, * gpu_res;
            cudaMalloc((void**)(&gpu_arr), size * sizeof(double));
            cudaMalloc((void**)(&gpu_res), size * sizeof(double));
            double alpha = 4.0;
            const int blockNum = (size + blockSize - 1) / blockSize;
            cudaMemcpy(gpu_arr, arr, size * sizeof(double), cudaMemcpyHostToDevice);
            double start = omp_get_wtime();
            daxpyKernelGpu << <blockNum, blockSize >> > (size, alpha, gpu_arr, 1, gpu_res, 1);
            cudaDeviceSynchronize();
            double finish = omp_get_wtime();
            cudaMemcpy(arr, gpu_res, size * sizeof(double), cudaMemcpyDeviceToHost);
            printf("block size = %d, time: %f\n", blockSize, (finish - start) * 1000);
            /*for (int i = 0; i < size; i++) {
                double c = i * alpha;
                if (c != arr[i]) {
                    printf("%f, %f\n", c, arr[i]);
                }
            }*/
            cudaFree(gpu_arr);
            cudaFree(gpu_res);
            delete[] arr;
        }
        printf("\n");
    }
}

void saxpyRun() {
    std::vector<int> sizes = { (int)1e6, (int)1e7, (int)1e8, (int)2e8 };
    for (auto size : sizes)
    {
        printf("size: %d\n", size);
        for (int blockSize = 8; blockSize <= 256; blockSize *= 2)
        {
            float* arr = new float[size];
            for (int i = 0; i < size; i++) {
                arr[i] = i;
            }
            float* gpu_arr, * gpu_res;
            cudaMalloc((void**)(&gpu_arr), size * sizeof(float));
            cudaMalloc((void**)(&gpu_res), size * sizeof(float));
            float alpha = 4.0;
            const int blockNum = (size + blockSize - 1) / blockSize;
            cudaMemcpy(gpu_arr, arr, size * sizeof(float), cudaMemcpyHostToDevice);
            double start = omp_get_wtime();
            saxpyKernelGpu << <blockNum, blockSize >> > (size, alpha, gpu_arr, 1, gpu_res, 1);
            cudaDeviceSynchronize();
            double finish = omp_get_wtime();
            cudaMemcpy(arr, gpu_res, size * sizeof(float), cudaMemcpyDeviceToHost);
            printf("block size = %d, time: %f\n", blockSize, (finish - start) * 1000);
            /*for (int i = 0; i < size; i++) {
                float c = i * alpha;
                if (c != arr[i]) {
                    printf("%f, %f\n", c, arr[i]);
                }
            }*/
            cudaFree(gpu_arr);
            cudaFree(gpu_res);
            delete[] arr;
        }
        printf("\n");
    }
}
