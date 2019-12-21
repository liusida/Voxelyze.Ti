#include <vector>
#include "TI_VoxelyzeKernel.h"
#include "TI_Link.h"

TI_Link::TI_Link(CVX_Link* p, TI_VoxelyzeKernel* k)
{
    _link = p;
    _kernel = k;

    pVNeg = getGPUPointer(p->pVNeg);
    pVPos = getGPUPointer(p->pVPos);

    strain = p->strain;
    pos2 = p->pos2;
}

TI_Voxel* TI_Link::getGPUPointer(CVX_Voxel* p) {
    //search host pointer in _kernel->h_voxels, get the index and get GPU pointer from _kernel->d_voxels.
    std::vector<CVX_Voxel *>::iterator it;
    it = find (_kernel->h_voxels.begin(), _kernel->h_voxels.end(), p);
    if (it != _kernel->h_voxels.end()) {
        int index = std::distance(_kernel->h_voxels.begin(), it);
        return _kernel->d_voxels[index];
    }
    else {
        std::cout << "ERROR: voxel for link not found. Maybe the input CVoxelyze* Vx is broken.\n";
    }
    return NULL;
}

CUDA_CALLABLE_MEMBER void TI_Link::test() {
    printf("test\n");
}

CUDA_CALLABLE_MEMBER void TI_Link::updateForces() {
    double a;
	//Vec3D<double> oldPos2 = pos2, oldAngle1v = angle1v, oldAngle2v = angle2v; //remember the positions/angles from last timestep to calculate velocity
    //orientLink();

    printf("Force updated.\n");
}
CUDA_CALLABLE_MEMBER Quat3D<double> TI_Link::orientLink(/*double restLength*/) //updates pos2, angle1, angle2, and smallAngle
{
    //pos2 = toAxisX(Vec3D<double>(pVPos->position() - pVNeg->position())); //digit truncation happens here...

    //Quat3D<double> totalRot;// = angle1.Conjugate(); //keep track of the total rotation of this bond (after toAxisX())

    //return totalRot;
    return NULL;
}
TI_Link::~TI_Link()
{
}
