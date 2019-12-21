#include <iostream>

#include "TI_VoxelyzeKernel.h"

TI_VoxelyzeKernel::TI_VoxelyzeKernel( CVoxelyze* vx )
{
    _vx = vx;
    for (auto link: vx->linksList) {
        _linksList.push_back(new TI_Link(link));
    }
}

TI_VoxelyzeKernel::~TI_VoxelyzeKernel()
{
}

__global__ void gpu_function_1(int* a, int num) {
    int gindex = threadIdx.x + blockIdx.x * blockDim.x; 
    if (gindex < num) {
        a[gindex] = gindex;
    }
}

void TI_VoxelyzeKernel::simpleGPUFunction() {
    int* d_a;
    int* a;
    int num = 10;
    int mem_size = num * sizeof(int);

    a = (int *) malloc(mem_size);
    cudaMalloc( &d_a, mem_size );

    gpu_function_1<<<1,num>>>(d_a, num);
    cudaMemcpy(a, d_a, mem_size, cudaMemcpyDeviceToHost);

    for (int i=0;i<num;i++) {
        std::cout<< a[i] << ",";
    }
    std::cout << std::endl;
}

void TI_VoxelyzeKernel::doTimeStep(double dt) {

}