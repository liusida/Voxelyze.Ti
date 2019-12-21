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
    void simpleGPUFunction();

    CVoxelyze* _vx;

    thrust::device_vector<TI_Link *> d_links;
    thrust::device_vector<TI_Voxel *> d_voxels;
    std::vector<CVX_Link *> h_links;
    std::vector<CVX_Voxel *> h_voxels;

    // h_links[i]  -- coresponding to -->  d_links[i]
    
};


#endif // TI_VOXELYZE_KERNEL_H
