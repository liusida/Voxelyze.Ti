#include <iostream>

#include "Voxelyze.h"
#include "TI_VoxelyzeKernel.h"

int main(int argc, char** argv) {

    int steps = 1000;
    if (argc>=2) {
        steps = strtol (argv[1],NULL,10);
        debugHost( printf("steps: %d", steps) );
    }

    CVoxelyze Vx(0.005); //5mm voxels

    Vx.enableFloor();
    Vx.enableCollisions();

    CVX_Material* pMaterial = Vx.addMaterial(1000000, 1000); //A material with stiffness E=1MPa and density 1000Kg/m^3
    // for (int i=0;i<3;i++) {
    //     for (int j=0;j<3;j++) {
    //         for (int k=0;k<3;k++) {
    //             Vx.setVoxel(pMaterial, i,j,k);
    //         }
    //     }
    // }
    
    CVX_Voxel* Voxel1 = Vx.setVoxel(pMaterial, 2, 0, 1); //Voxel at index x=0, y=0. z=0
    // CVX_Voxel* Voxel2 = Vx.setVoxel(pMaterial, 1, 0, 4);
    CVX_Voxel* Voxel3 = Vx.setVoxel(pMaterial, 2, 0, 5); //Beam extends in the +X direction

    // for (unsigned i=0;i<3000;i++) {
    //     Vx.setVoxel(pMaterial, -1,-1, i);
    // }

    // Voxel1->external()->setFixedAll(); //Fixes all 6 degrees of freedom with an external condition on Voxel 1
    Voxel3->external()->setForce(0, 0, -1); //pulls Voxel 3 downward with 1 Newton of force.

    TI_VoxelyzeKernel VxKernel(&Vx);
    VxKernel.readVoxelsPosFromDev();
    
    for (int j=0;j<steps;j++) {

        VxKernel.doTimeStep(0.00001);
        
        bool ret = Vx.doTimeStep(0.00001);
        if (!ret) {
            debugHost( printf("ERROR: Vx doTimeStep return false!") );
            break;
        }

        //if (j>=184 && j%1==0 && j<186) {
        if (j%100==0) {
            debugHostx("step", printf("%d", j));
            VxKernel.readVoxelsPosFromDev();
            for (unsigned i=0;i<VxKernel.read_voxels.size();i++) {
                TI_Voxel* temp = VxKernel.read_voxels[i];
                debugDev( printf("[%d] Dev Position: %f, %f, %f.", i, temp->pos.x, temp->pos.y, temp->pos.z) );
        
            }
            for (unsigned i=0;i<Vx.voxelCount();i++) {
                CVX_Voxel* temp = Vx.voxel(i);
                debugHost( printf("[%d] Host Position: %f, %f, %f.", i, temp->pos.x, temp->pos.y, temp->pos.z) );
            }
        }
    }


    std::cout<<std::endl;
}
