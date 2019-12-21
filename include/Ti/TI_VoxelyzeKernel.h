#if !defined(TI_VOXELYZE_KERNEL_H)
#define TI_VOXELYZE_KERNEL_H

#include "Voxelyze.h"

class CTI_VoxelyzeKernel
{
private:
    /* data */
public:
    CTI_VoxelyzeKernel( CVoxelyze* vx );
    ~CTI_VoxelyzeKernel();

    void doTimeStep(double dt=0.001f);
    void simpleGPUFunction();
    
    CVoxelyze* _vx;
};


#endif // TI_VOXELYZE_KERNEL_H
