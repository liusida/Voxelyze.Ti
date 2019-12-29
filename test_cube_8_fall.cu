#include <iostream>

#include "Voxelyze.h"
#include "TI_VoxelyzeKernel.h"

int main(int argc, char** argv) {

    int steps = 5000;
    if (argc>=2) {
        steps = strtol (argv[1],NULL,10);
        debugHost( printf("steps: %d", steps) );
    }

    CVoxelyze Vx(0.005); //5mm voxels

    
    Vx.enableFloor();
    Vx.enableCollisions();
    Vx.setGravity();

    CVX_Material* pMaterial = Vx.addMaterial(1000000, 1000); //A material with stiffness E=1MPa and density 1000Kg/m^3


    for (int i=0;i<8;i++) {
        for (int j=0;j<8;j++) {
            for (int k=0;k<8;k++) {
                CVX_Voxel* v = Vx.setVoxel(pMaterial, i,j,k+1);
            }
        }
    }
    // for (int i=0;i<2;i++) {
    //     for (int j=0;j<2;j++) {
    //         for (int k=0;k<2;k++) {
    //             CVX_Voxel* v = Vx.setVoxel(pMaterial, i,j,k+1);
    //         }
    //     }
    // }
    for (unsigned i=0;i<3192;i++)
        Vx.doTimeStep(0.00001);

    TI_VoxelyzeKernel VxKernel(&Vx);
    //VxKernel.readVoxelsPosFromDev();
    #define BYTE_TO_BINARY_PATTERN "%c%c%c%c%c%c%c%c"
    #define BYTE_TO_BINARY(byte)  \
      (byte & 0x80 ? '1' : '0'), \
      (byte & 0x40 ? '1' : '0'), \
      (byte & 0x20 ? '1' : '0'), \
      (byte & 0x10 ? '1' : '0'), \
      (byte & 0x08 ? '1' : '0'), \
      (byte & 0x04 ? '1' : '0'), \
      (byte & 0x02 ? '1' : '0'), \
      (byte & 0x01 ? '1' : '0') 
    
    for (int j=0;j<5;j++) {

        VxKernel.doTimeStep(0.00001);
        
        bool ret = Vx.doTimeStep(0.00001);
        if (!ret) {
            debugHost( printf("ERROR: Vx doTimeStep return false!") );
            break;
        }

        debugHostx("step", printf("%d", j));
        VxKernel.readVoxelsPosFromDev();
        for (unsigned i=0;i<VxKernel.read_voxels.size();i++) {
            TI_Voxel* temp = VxKernel.read_voxels[i];
            if (temp->ix==0 && temp->iy==0 && temp->iz==1) {
                debugDev( printf("[%d] Dev Position: %e, %e, %e. linMom: %e %e %e. boolStates %d("BYTE_TO_BINARY_PATTERN").", i, temp->pos.x, temp->pos.y, temp->pos.z, temp->linMom.x, temp->linMom.y, temp->linMom.z, temp->boolStates, BYTE_TO_BINARY(temp->boolStates)) );
                
            }
            if (temp->ix==1 && temp->iy==1 && temp->iz==2) {
                debugDev( printf("[%d] Dev Position: %e, %e, %e. linMom: %e %e %e. boolStates %d("BYTE_TO_BINARY_PATTERN").", i, temp->pos.x, temp->pos.y, temp->pos.z, temp->linMom.x, temp->linMom.y, temp->linMom.z, temp->boolStates, BYTE_TO_BINARY(temp->boolStates)) );
                
            }
    
        }
        for (unsigned i=0;i<Vx.voxelCount();i++) {
            CVX_Voxel* temp = Vx.voxel(i);
            if (temp->ix==0 && temp->iy==0 && temp->iz==1) {
                debugHost( printf("[%d] Host Position: %e, %e, %e. linMom: %e %e %e. boolStates %d("BYTE_TO_BINARY_PATTERN").", i, temp->pos.x, temp->pos.y, temp->pos.z, temp->linMom.x, temp->linMom.y, temp->linMom.z, temp->boolStates, BYTE_TO_BINARY(temp->boolStates)) );
                
            }
            if (temp->ix==1 && temp->iy==1 && temp->iz==2) {
                debugHost( printf("[%d] Host Position: %e, %e, %e. linMom: %e %e %e. boolStates %d("BYTE_TO_BINARY_PATTERN").", i, temp->pos.x, temp->pos.y, temp->pos.z, temp->linMom.x, temp->linMom.y, temp->linMom.z, temp->boolStates, BYTE_TO_BINARY(temp->boolStates)) );
                
            }
        }
    }

    std::cout<<std::endl;
}
