#include <cuda_runtime.h>
#include <cstdio>

__global__ void smoke_kernel(float* x) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < 1024) x[i] = x[i] * 2.0f + 1.0f;
}

int main() {
    float* d = nullptr;
    cudaError_t err = cudaMalloc(&d, 1024 * sizeof(float));
    if (err != cudaSuccess) {
        std::fprintf(stderr, "cudaMalloc: %s\n", cudaGetErrorString(err));
        return 1;
    }

    err = cudaMemset(d, 0, 1024 * sizeof(float));
    if (err != cudaSuccess) {
        std::fprintf(stderr, "cudaMemset: %s\n", cudaGetErrorString(err));
        cudaFree(d);
        return 1;
    }

    smoke_kernel<<<4, 256>>>(d);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::fprintf(stderr, "launch: %s\n", cudaGetErrorString(err));
        cudaFree(d);
        return 1;
    }

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::fprintf(stderr, "sync: %s\n", cudaGetErrorString(err));
        cudaFree(d);
        return 1;
    }

    float h[1024] = {};
    err = cudaMemcpy(h, d, sizeof(h), cudaMemcpyDeviceToHost);
    cudaFree(d);
    if (err != cudaSuccess) {
        std::fprintf(stderr, "cudaMemcpy: %s\n", cudaGetErrorString(err));
        return 1;
    }

    std::printf("cuda_smoke_result=%0.1f\n", h[0]);
    return h[0] == 1.0f ? 0 : 2;
}
