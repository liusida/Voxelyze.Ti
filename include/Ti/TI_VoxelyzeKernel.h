#if !defined(TI_VOXELYZE_KERNEL_H)
#define TI_VOXELYZE_KERNEL_H

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
    std::vector<TI_Link* > _linksList;
};


#endif // TI_VOXELYZE_KERNEL_H
