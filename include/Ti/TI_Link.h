#if !defined(TI_LINK_H)
#define TI_LINK_H

#include "VX_Link.h"
#include "Vec3D.h"
#include "TI_Utils.h"

class TI_VoxelyzeKernel;
class TI_Voxel;

enum linkAxis {			
    X_AXIS = 0,			//!< X Axis
    Y_AXIS = 1,			//!< Y Axis
    Z_AXIS = 2			//!< Z Axis
};

class TI_Link
{
public:
    TI_Link(CVX_Link* p, TI_VoxelyzeKernel* k);
    ~TI_Link();

    TI_Voxel* getGPUPointer(CVX_Voxel* p);

    CUDA_CALLABLE_MEMBER void test();
    CUDA_CALLABLE_MEMBER void updateForces();
	CUDA_CALLABLE_MEMBER Quat3D<double> orientLink(/*double restLength*/); //updates pos2, angle1, angle2, and smallAngle. returns the rotation quaternion (after toAxisX) used to get to this orientation

	template <typename T>
    CUDA_CALLABLE_MEMBER Vec3D<T> toAxisX		(const Vec3D<T>& v)	const {switch (axis){case Y_AXIS: return Vec3D<T>(v.y, -v.x, v.z); case Z_AXIS: return Vec3D<T>(v.z, v.y, -v.x); default: return v;}} //transforms a vec3D in the original orientation of the bond to that as if the bond was in +X direction

/*data*/

    CVX_Link* _link;
    TI_VoxelyzeKernel* _kernel;

    TI_Voxel *pVNeg, *pVPos;
	Vec3D<> forceNeg, forcePos;
	Vec3D<> momentNeg, momentPos;

	float strain;
	float maxStrain, /*maxStrainRatio,*/ strainOffset; //keep track of the maximums for yield/fail/nonlinear materials (and the ratio of the maximum from 0 to 1 [all positive end strain to all negative end strain])

	linkAxis axis;

	float strainRatio; //ration of Epos to Eneg (EPos/Eneg)

	Vec3D<double> pos2, angle1v, angle2v; //pos1 is always = 0,0,0
	Quat3D<double> angle1, angle2; //this bond in local coordinates. 
	bool smallAngle; //based on compiled precision setting
	double currentRestLength;
	float currentTransverseArea, currentTransverseStrainSum; //so we don't have to re-calculate everytime

	float _stress; //keep this around for convenience

};



#endif // TI_LINK_H
