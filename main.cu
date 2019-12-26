#include <iostream>

#include "Voxelyze.h"
#include "TI_VoxelyzeKernel.h"

int main(int argc, char** argv) {

    int steps = 2;
    if (argc>=2) {
        steps = strtol (argv[1],NULL,10);
        debugHost( printf("steps: %d", steps) );
    }

    CVoxelyze Vx(0.005); //5mm voxels

    Vx.enableFloor();

    CVX_Material* pMaterial = Vx.addMaterial(1000000, 1000); //A material with stiffness E=1MPa and density 1000Kg/m^3
    CVX_Voxel* Voxel1 = Vx.setVoxel(pMaterial, 0, 0, 0); //Voxel at index x=0, y=0. z=0
    CVX_Voxel* Voxel2 = Vx.setVoxel(pMaterial, 1, 0, 0);
    CVX_Voxel* Voxel3 = Vx.setVoxel(pMaterial, 2, 0, 0); //Beam extends in the +X direction

    for (unsigned i=0;i<3000;i++) {
        Vx.setVoxel(pMaterial, -1,-1, i);
    }

    //Voxel1->external()->setFixedAll(); //Fixes all 6 degrees of freedom with an external condition on Voxel 1
    Voxel3->external()->setForce(0, 0, 1); //pulls Voxel 3 downward with 1 Newton of force.

    TI_VoxelyzeKernel VxKernel(&Vx);
    VxKernel.readVoxelsPosFromDev();
    
    for (int i=0;i<steps;i++) {
        //debugHostx("step", printf("%d", i));

        VxKernel.doTimeStep(0.00001);
        
        // bool ret = Vx.doTimeStep(0.00001);
        // if (!ret) {
        //     debugHost( printf("ERROR: Vx doTimeStep return false!") );
        //     break;
        // }
    }
    VxKernel.readVoxelsPosFromDev();
    // for (unsigned i=0;i<VxKernel.read_voxels.size();i++) {
    //     TI_Voxel* temp = VxKernel.read_voxels[i];
    //     debugDev( printf("[%d] Dev Position: %f, %f, %f.", i, temp->pos.x, temp->pos.y, temp->pos.z) );

    // }
    // for (unsigned i=0;i<Vx.voxelCount();i++) {
    //     CVX_Voxel* temp = Vx.voxel(i);
    //     debugHost( printf("[%d] Host Position: %f, %f, %f.", i, temp->pos.x, temp->pos.y, temp->pos.z) );
    // }

    std::cout<<std::endl;
}
