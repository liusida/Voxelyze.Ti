#if !defined(TI_VOXELYZE_KERNEL_H)
#define TI_VOXELYZE_KERNEL_H

#include <thrust/device_vector.h>

#include "TI_Link.h"
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
};


#endif // TI_VOXELYZE_KERNEL_H
