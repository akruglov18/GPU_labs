
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda_device_runtime_api.h>
#include <omp.h>
#include <stdio.h>
#include <random>

#define BLOCK_SIZE 16

__global__ void matmulKernel(int* a, int* b, int* c, int n)
{
    int sum = 0;
    int global_row = blockIdx.y * blockDim.y + threadIdx.y;
    int global_col = blockIdx.x * blockDim.x + threadIdx.x;

    for (int i = 0; i < n; i++)
    {
        sum += a[global_row * n + i] * b[i * n + global_col];
    }
    c[global_row * n + global_col] = sum;
}

__global__ void matmulKernelOpt(int* a, int* b, int* c, int n)
{
    int sum = 0;
    int global_row = blockIdx.y * blockDim.y + threadIdx.y;
    int global_col = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ int A[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ int B[BLOCK_SIZE][BLOCK_SIZE];

    for (int i = 0; i < n / BLOCK_SIZE; i++)
    {
        int col = i * BLOCK_SIZE + threadIdx.x;
        int row = i * BLOCK_SIZE + threadIdx.y;

        A[threadIdx.y][threadIdx.x] = a[global_row * n + col];
        B[threadIdx.y][threadIdx.x] = b[row * n + global_col];

        __syncthreads();
        for (int j = 0; j < BLOCK_SIZE; j++) {
            sum += A[threadIdx.y][j] * B[j][threadIdx.x];
        }
        __syncthreads();
    }
    c[global_row * n + global_col] = sum;
}

void printMatrix(int* A, int n) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            printf("%d ", A[i * n + j]);
        }
        printf("\n");
    }
}

void validate(int* a, int* b, int* ref, int N);

void runKernel();
void runOptKernel();

std::vector<int> sizes = { 512, 1056, 1520, 2064, 3024 };

int main()
{
    printf("Kernel:\n");
    runKernel();
    printf("------------------------\n");
    printf("Optimized Kernel:\n");
    runOptKernel();
}

void runKernel() {
    for (const int N : sizes)
    {
        if (N % BLOCK_SIZE != 0) {
            printf("Invalid size: %d\n", N);
            continue;
        }
        int* a = new int[N * N];
        int* b = new int[N * N];
        int* res = new int[N * N];
        for (int i = 0; i < N * N; i++) {
            a[i] = rand() % 20;
            b[i] = rand() % 20;
        }
        int* a_gpu, * b_gpu, * c_gpu;
        cudaMalloc((void**)(&a_gpu), sizeof(int) * N * N);
        cudaMalloc((void**)(&b_gpu), sizeof(int) * N * N);
        cudaMalloc((void**)(&c_gpu), sizeof(int) * N * N);
        cudaMemcpy(a_gpu, a, sizeof(int) * N * N, cudaMemcpyHostToDevice);
        cudaMemcpy(b_gpu, b, sizeof(int) * N * N, cudaMemcpyHostToDevice);
        dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
        dim3 blocks(N / threads.x, N / threads.y);
        double time = omp_get_wtime();
        matmulKernel << <blocks, threads >> > (a_gpu, b_gpu, c_gpu, N);
        cudaDeviceSynchronize();
        time = omp_get_wtime() - time;
        printf("%f\n", time);
        cudaMemcpy(res, c_gpu, sizeof(int) * N * N, cudaMemcpyDeviceToHost);
        //printMatrix(a, N);
        //printMatrix(b, N);
        //printMatrix(res, N);
        cudaFree(a_gpu);
        cudaFree(b_gpu);
        cudaFree(c_gpu);
        //validate(a, b, res, N);
        delete[] a;
        delete[] b;
        delete[] res;
    }
}

void runOptKernel() {
    for (const int N : sizes)
    {
        if (N % BLOCK_SIZE != 0) {
            printf("Invalid size: %d\n", N);
            continue;
        }
        int* a = new int[N * N];
        int* b = new int[N * N];
        int* res = new int[N * N];
        for (int i = 0; i < N * N; i++) {
            a[i] = b[i] = rand() % 20;
        }
        int* a_gpu, * b_gpu, * c_gpu;
        cudaMalloc((void**)(&a_gpu), sizeof(int) * N * N);
        cudaMalloc((void**)(&b_gpu), sizeof(int) * N * N);
        cudaMalloc((void**)(&c_gpu), sizeof(int) * N * N);
        cudaMemcpy(a_gpu, a, sizeof(int) * N * N, cudaMemcpyHostToDevice);
        cudaMemcpy(b_gpu, b, sizeof(int) * N * N, cudaMemcpyHostToDevice);
        dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
        dim3 blocks(N / threads.x, N / threads.y);
        double time = omp_get_wtime();
        matmulKernelOpt << <blocks, threads >> > (a_gpu, b_gpu, c_gpu, N);
        cudaDeviceSynchronize();
        time = omp_get_wtime() - time;
        printf("%f\n", time);
        cudaMemcpy(res, c_gpu, sizeof(int) * N * N, cudaMemcpyDeviceToHost);
        //printMatrix(a, N);
        //printMatrix(b, N);
        //printMatrix(res, N);
        cudaFree(a_gpu);
        cudaFree(b_gpu);
        cudaFree(c_gpu);
        //validate(a, b, res, N);
        delete[] a;
        delete[] b;
        delete[] res;
    }
}

void get(int* a, int* b, int* c, int N)
{
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            int sum = 0;
            for (int k = 0; k < N; k++) {
                sum += a[i * N + k] * b[j + k * N];
            }
            c[i * N + j] = sum;
        }
    }
}

void validate(int* a, int* b, int* ref, int N) {
    int* c = new int[N * N];
    get(a, b, c, N);
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            if (ref[i * N + j] != c[i * N + j])
            {
                printf("%d ", c[i * N + j]);
            }
        }
        //printf("\n");
    }
    delete[] c;
}
