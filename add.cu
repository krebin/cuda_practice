#include <iostream>

__global__
void vecAddKernel(float *A, float *B, float *C, int n)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if(i < n)
    {
        C[i] = A[i] + B[i];
    }
}

void vecAdd(float* h_A, float* h_B, float* h_C, int n)
{
    int size = n * sizeof(float);
    float *d_A, *d_B, *d_C;
    cudaMalloc((void **) &d_A, size);
    cudaMalloc((void **) &d_B, size);
    cudaMalloc((void **) &d_C, size);

    cudaError_t err = cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);

    if (err != cudaSuccess)
    {
        printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__,__LINE__);
        exit(EXIT_FAILURE);
    }


    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    vecAddKernel<<<ceil(n/256.0), 256>>>(d_A, d_B, d_C, n);

    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

int main()
{
    float A[] = {1, 2, 3, 4};
    float B[] = {1, 2, 3, 4};
    float C[4] = {};
    vecAdd(A, B, C, 4);

    for(float i : A)
    {
        std::cout << i << " ";
    }
    std::cout << "\n";

    for(float i : B)
    {
        std::cout << i << " ";
    }
    std::cout << "\n";

    for(float i : C)
    {
        std::cout << i << " ";
    }
    std::cout << "\n";

}
