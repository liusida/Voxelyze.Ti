#if !defined(TI_VOXELYZE_KERNEL_H)
#define TI_VOXELYZE_KERNEL_H

#include <thrust/device_vector.h>
#include <vector>

#include "TI_Link.h"
#include "TI_Voxel.h"
#include "Voxelyze.h"

class TI_VoxelyzeKernel
{
private:
    /* data */
public:
    TI_VoxelyzeKernel( CVoxelyze* vx );
    ~TI_VoxelyzeKernel();

    void doTimeStep(double dt=0.001f);

    void updateCollisions();
    void regenerateCollisions(double threshRadiusSq);
    void clearCollisions();

    void simpleGPUFunction();

    void readVoxelsPosFromDev(); //read only position of voxels.

    CVoxelyze* _vx;

    thrust::device_vector<TI_Link *> d_links;
    thrust::device_vector<TI_Voxel *> d_voxels;
    std::vector<CVX_Link *> h_links;
    std::vector<CVX_Voxel *> h_voxels;

    std::vector<TI_Link *> read_links;
    std::vector<TI_Voxel *> read_voxels;

    TI_vector<TI_Collision *>* d_collisions;
    TI_vector<TI_Collision *> h_collisions;
    
    // h_links[i]  -- coresponding to -->  d_links[i]
	float currentTime; //current time of the simulation in seconds

    bool nearbyStale;
    bool collisionsStale;
    bool* d_collisionsStale;

    TI_Voxel** cached_d_voxels;
    int cached_num_d_voxels;
    int cached_gridSize_voxels;
};


#endif // TI_VOXELYZE_KERNEL_H
