#include <iostream>

#include "TI_VoxelyzeKernel.h"
#include "TI_Utils.h"

TI_VoxelyzeKernel::TI_VoxelyzeKernel( CVoxelyze* vx )
{
    _vx = vx;
    for (auto link: vx->linksList) {
        //alloc a GPU memory space
        TI_Link * d_link;
        gpuErrchk(cudaMalloc((void **) &d_link, sizeof(TI_Link)));
        //set values for GPU memory space
        TI_Link temp = TI_Link(link);
        gpuErrchk(cudaMemcpy(d_link, &temp, sizeof(TI_Link), cudaMemcpyHostToDevice));
        //save the pointer
        d_links.push_back(d_link);
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

__global__ void gpu_update_force(TI_Link** links, int num) {
    int gindex = threadIdx.x + blockIdx.x * blockDim.x; 
    if (gindex < num) {
        //TODO: update force for links[gindex];
        TI_Link* t = links[gindex];
        printf("GPU strain: %f\n", t->strain);
    }
}
void TI_VoxelyzeKernel::doTimeStep(double dt) {
    int blockSize = 256;
    int N = d_links.size();
    int gridSize = (N + blockSize - 1) / blockSize;
    gpu_update_force<<<gridSize, blockSize>>>(thrust::raw_pointer_cast(d_links.data()), N);
}