#include <stdio.h>

#include "TI_Object.h"

__global__ void kernel(float *A, unsigned num_A) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i<num_A) {
        A[i] = i;
    }
}
CTI_Object::CTI_Object() {
    
}
void CTI_Object::Try() {
    float* d_A;
    float* h_A;
    unsigned num_A = 1024;
    unsigned size = num_A*num_A * sizeof(float);
    cudaMalloc(&d_A, size);
    kernel<<<1024,1024>>>(d_A, num_A*num_A);
    h_A = (float *)malloc( size );
    cudaMemcpy(h_A, d_A, size, cudaMemcpyDeviceToHost );
    cudaFree(d_A);
    for(unsigned i=num_A*4;i<num_A*4+5;i++) {
        printf("%f, ", h_A[i]);
    }
    printf("\n");
    delete h_A;
}