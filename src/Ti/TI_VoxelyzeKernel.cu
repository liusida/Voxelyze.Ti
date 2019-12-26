#include <iostream>
#include "cuda_occupancy.h"

#include "TI_VoxelyzeKernel.h"
#include "TI_Utils.h"
TI_VoxelyzeKernel::TI_VoxelyzeKernel( CVoxelyze* vx ):
currentTime(vx->currentTime), nearbyStale(true), collisionsStale(true)
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
        //ebugDevice("t->axialStrain()", printf("%f", t->axialStrain()));
        if (t->axialStrain() > 100) { printf("ERROR: Diverged."); }
    }
}
__global__
void gpu_update_voxel(TI_Voxel** voxels, int num, double dt) {
    int gindex = threadIdx.x + blockIdx.x * blockDim.x; 
    if (gindex < num) {
        TI_Voxel* t = voxels[gindex];
        t->timeStep(dt);
    }
}
__global__
void generate_voxels_Nearby(TI_Voxel** voxels, int num, float watchRadiusVx) {
    int gindex = threadIdx.x + blockIdx.x * blockDim.x; 
    if (gindex < num) {
        TI_Voxel* t = voxels[gindex];
        t->generateNearby(watchRadiusVx*2, false);
    }
}
__global__
void gpu_update_contact_force(TI_Collision** collisions, int num) {
    int gindex = threadIdx.x + blockIdx.x * blockDim.x; 
    if (gindex < num) {
        TI_Collision* t = collisions[gindex];
        t->updateContactForce();
    }
}
__global__
void gpu_clear_collision(TI_Voxel** voxels, int num) {
    int gindex = threadIdx.x + blockIdx.x * blockDim.x; 
    if (gindex < num) {
        TI_Voxel* t = voxels[gindex];
        t->colWatch.clear();
    }
}
__global__
void gpu_generate_collision(TI_Voxel** voxels, int num) {

}
void TI_VoxelyzeKernel::clearCollisions() {
    for (unsigned i=0;i<d_collisions.size();i++) {
        cudaFree(d_collisions[i]);
    }
    d_collisions.clear();
    int blockSize = 1024;
    int num_voxels = d_voxels.size();
    int gridSize_voxels = (num_voxels + blockSize - 1) / blockSize; 
    gpu_clear_collision<<<gridSize_voxels, blockSize>>>(thrust::raw_pointer_cast(d_voxels.data()), num_voxels);
    cudaDeviceSynchronize();
}
void TI_VoxelyzeKernel::regenerateCollisions(float threshRadiusSq) {
    clearCollisions();
    //TODO: regenerate collisions
    //gpu_generate_collision<<<>>>()


}
void TI_VoxelyzeKernel::updateCollisions() {
    int blockSize = 1024;
    int num_voxels = d_voxels.size();
    int gridSize_voxels = (num_voxels + blockSize - 1) / blockSize; 
    float watchRadiusVx = 2*_vx->boundingRadius + _vx->watchDistance; //outer radius to track all voxels within
	float watchRadiusMm = (float)(_vx->voxSize * watchRadiusVx); //outer radius to track all voxels within
	float recalcDist = (float)(_vx->voxSize * _vx->watchDistance / 2 ); //if the voxel moves further than this radius, recalc! //1/2 the allowabl, accounting for 0.5x radius of the voxel iself

    //TODO: should decide when to update
	if (nearbyStale){
        generate_voxels_Nearby<<<gridSize_voxels, blockSize>>>(thrust::raw_pointer_cast(d_voxels.data()), num_voxels, watchRadiusVx);
        cudaDeviceSynchronize();
		nearbyStale = false;
		collisionsStale = true;
    }
    if (collisionsStale){
        regenerateCollisions(watchRadiusMm*watchRadiusMm);
        collisionsStale = false;
    }

    //check if any voxels have moved far enough to make collisions stale
    int num_collisions = d_collisions.size();
    int gridSize_collisions = (num_collisions + blockSize - 1) / blockSize; 
    gpu_update_contact_force<<<gridSize_collisions, blockSize>>>(thrust::raw_pointer_cast(d_collisions.data()), num_collisions);

}

void TI_VoxelyzeKernel::doTimeStep(double dt) {
    int blockSize = 1024;
    int num_links = d_links.size();
    int num_voxels = d_voxels.size();
    int gridSize_links = (num_links + blockSize - 1) / blockSize; 
    int gridSize_voxels = (num_voxels + blockSize - 1) / blockSize; 
    gpu_update_force<<<gridSize_links, blockSize>>>(thrust::raw_pointer_cast(d_links.data()), num_links);
    cudaDeviceSynchronize();
    
    updateCollisions();
    cudaDeviceSynchronize();

    gpu_update_voxel<<<gridSize_voxels, blockSize>>>(thrust::raw_pointer_cast(d_voxels.data()), num_voxels, dt);
    cudaDeviceSynchronize();

    currentTime += dt;
}

void TI_VoxelyzeKernel::readVoxelsPosFromDev() {
    for (auto l:read_links) delete l;
    for (auto v:read_voxels) delete v;
    read_links.clear();
    read_voxels.clear();

    for (unsigned i=0;i<d_voxels.size();i++) {
        TI_Voxel* temp = (TI_Voxel*) malloc(sizeof(TI_Voxel));
        cudaMemcpy(temp, d_voxels[i], sizeof(TI_Voxel), cudaMemcpyDeviceToHost);
        read_voxels.push_back(temp);
    }
    for (unsigned i=0;i<d_links.size();i++) {
        TI_Link* temp = (TI_Link*) malloc(sizeof(TI_Link));
        cudaMemcpy(temp, d_links[i], sizeof(TI_Link), cudaMemcpyDeviceToHost);
        read_links.push_back(temp);
    }
}