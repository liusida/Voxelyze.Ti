#if !defined(TI_VOXEL_H)
#define TI_VOXEL_H

#include "VX_Voxel.h"
#include "TI_Utils.h"

class TI_Voxel
{
public:
    TI_Voxel(CVX_Voxel* p);

	CUDA_CALLABLE_MEMBER Vec3D<double> position() const {return pos;} //!< Returns the center position of this voxel in meters (GCS). This is the origin of the local coordinate system (LCS).

/* data */
    CVX_Voxel* _voxel;
    
    Vec3D<double> pos;

};

#endif // TI_VOXEL_H
