#include <iostream>

#include "Voxelyze.h"
#include "TI_VoxelyzeKernel.h"

int main(int, char**) {
    CVoxelyze Vx(0.005); //5mm voxels
    CVX_Material* pMaterial = Vx.addMaterial(1000000, 1000); //A material with stiffness E=1MPa and density 1000Kg/m^3
    CVX_Voxel* Voxel1 = Vx.setVoxel(pMaterial, 0, 0, 0); //Voxel at index x=0, y=0. z=0
    CVX_Voxel* Voxel2 = Vx.setVoxel(pMaterial, 1, 0, 0);
    CVX_Voxel* Voxel3 = Vx.setVoxel(pMaterial, 2, 0, 0); //Beam extends in the +X direction

    // for (unsigned i=0;i<1e3;i++) {
    //     Vx.setVoxel(pMaterial, -1,-1, i);
    // }

    Voxel1->external()->setFixedAll(); //Fixes all 6 degrees of freedom with an external condition on Voxel 1
    Voxel3->external()->setForce(1, 0, 0); //pulls Voxel 3 downward with 1 Newton of force.

    Vx.doTimeStep(0.0001);
    Vx.doTimeStep(0.0001);
    Vx.doTimeStep(0.0001);
    // for (int i=0;i<2;i++) {
    //     auto t = Vx.link(i);
    //     printf("Host pos2: %f, %f, %f\n", t->pos2.x, t->pos2.y, t->pos2.z);
    //     printf("HOST pVPos pos: %f, %f, %f\n", t->pVPos->pos.x, t->pVPos->pos.y, t->pVPos->pos.z );
    // }
    TI_VoxelyzeKernel VxKernel(&Vx);
    //VxKernel.simpleGPUFunction();

    VxKernel.doTimeStep(0.0001);
    Vx.doTimeStep(0.0001);

    std::cout<<std::endl;
}
