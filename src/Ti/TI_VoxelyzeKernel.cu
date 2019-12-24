#include <iostream>

#include "TI_VoxelyzeKernel.h"
#include "TI_Utils.h"

TI_VoxelyzeKernel::TI_VoxelyzeKernel( CVoxelyze* vx ):
currentTime(vx->currentTime)
{
    _vx = vx;
    
    //need to allocate memory first, then set the value, because there are links in voxels and voxels in links.

    for (auto voxel: vx->voxelsList) {
        //alloc a GPU memory space
        TI_Voxel * d_voxel;
        gpuErrchk(cudaMalloc((void **) &d_voxel, sizeof(TI_Voxel)));
        //save the pointer
        d_voxels.push_back(d_voxel);
        //save host pointer as well
        h_voxels.push_back(voxel);
    }
    for (auto link: vx->linksList) {
        //alloc a GPU memory space
        TI_Link * d_link;
        gpuErrchk(cudaMalloc((void **) &d_link, sizeof(TI_Link)));
        //set values for GPU memory space
        TI_Link temp = TI_Link(link, this);
        gpuErrchk(cudaMemcpy(d_link, &temp, sizeof(TI_Link), cudaMemcpyHostToDevice));
        //save the pointer
        d_links.push_back(d_link);
        //save host pointer as well
        h_links.push_back(link);
    }
    for (unsigned i=0;i<vx->voxelsList.size();i++) {
        TI_Voxel * d_voxel = d_voxels[i];
        CVX_Voxel * voxel = vx->voxelsList[i];
        //set values for GPU memory space
        TI_Voxel temp(voxel, this);
        gpuErrchk(cudaMemcpy(d_voxel, &temp, sizeof(TI_Voxel), cudaMemcpyHostToDevice));
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

__global__
void gpu_update_force(TI_Link** links, int num) {
    int gindex = threadIdx.x + blockIdx.x * blockDim.x; 
    if (gindex < num) {
        TI_Link* t = links[gindex];
        t->updateForces();
        debugDev();
        if (t->axialStrain() > 100) { printf("ERROR: Diverged."); }
        debugDev();
    }
}
__global__
void gpu_update_voxel(TI_Voxel** voxels, int num, double dt) {
    int gindex = threadIdx.x + blockIdx.x * blockDim.x; 
    if (gindex < num) {
        debugDev();
        TI_Voxel* t = voxels[gindex];
        t->timeStep(dt);
    }
}
void TI_VoxelyzeKernel::doTimeStep(double dt) {
    int blockSize = 256;
    int N = d_links.size();
    int gridSize = (N + blockSize - 1) / blockSize;
    gpu_update_force<<<gridSize, blockSize>>>(thrust::raw_pointer_cast(d_links.data()), N);
    cudaDeviceSynchronize();
    
    //TODO:updateCollision
    debugHost( printf("TODO:updateCollision") );
    //gpu_update_voxel
    blockSize = 256;
    N = d_voxels.size();
    gridSize = (N + blockSize - 1) / blockSize;
    gpu_update_voxel<<<gridSize, blockSize>>>(thrust::raw_pointer_cast(d_voxels.data()), N, dt);
    cudaDeviceSynchronize();

    currentTime += dt;
}