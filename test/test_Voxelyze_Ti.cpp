#include "gtest/gtest.h"
#ifdef _0
#include <thrust/device_vector.h>
#include "cuda.h"
#include "cuda_runtime.h"

#include "Voxelyze.h"
#include "TI_VoxelyzeKernel.h"

TEST(Voxelyze_Ti, throw_the_blob) {
    CVoxelyze Vx(0.005);
    Vx.enableFloor();
    Vx.enableCollisions();
    Vx.setGravity();

    CVX_Material* pMaterial = Vx.addMaterial(1000000, 1000); //A material with stiffness E=1MPa and density 1000Kg/m^3
    CVX_Voxel* v = Vx.setVoxel(pMaterial, 0,0,1);

    double time_step = Vx.recommendedTimeStep();

    v->external()->setForce(Vec3D<float>(1,2,0));
    for (int i=0;i<100;i++) {
        if (!Vx.doTimeStep(time_step)) {
            debugHost( printf("ERROR: Vx doTimeStep return false!") );break;
        }
    }
    v->external()->setForce(Vec3D<float>(0,0,0));

    TI_VoxelyzeKernel VxKernel(&Vx);
    for (int i=0;i<10000;i++) {
        VxKernel.doTimeStep(time_step);
        if (!Vx.doTimeStep(time_step)) {
            debugHost( printf("ERROR: Vx doTimeStep return false!") );break;
        }
    }
    VxKernel.readVoxelsPosFromDev();
    printf("VxKernel.read_voxels[0]->pos.x %f\n",VxKernel.read_voxels[0]->pos.x);
    printf("(*Vx.voxelList())[0]->pos.x %f\n",(*Vx.voxelList())[0]->pos.x);
    printf("VxKernel.read_voxels[0]->pos.z %f\n",VxKernel.read_voxels[0]->pos.z);
    printf("(*Vx.voxelList())[0]->pos.z %f\n",(*Vx.voxelList())[0]->pos.z);
    float difference_in_x = (VxKernel.read_voxels[0]->pos.x - (*Vx.voxelList())[0]->pos.x);
    float difference_in_y = (VxKernel.read_voxels[0]->pos.y - (*Vx.voxelList())[0]->pos.y);
    float difference_in_z = (VxKernel.read_voxels[0]->pos.z - (*Vx.voxelList())[0]->pos.z);
    EXPECT_LT(difference_in_x,0.01);
    EXPECT_LT(difference_in_y,0.01);
    EXPECT_LT(difference_in_z,0.01);
}
#endif