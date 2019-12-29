#include <cstdlib>

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
    for (int i=0;i<2;i++) {
        for (int j=0;j<2;j++) {
            for (int k=0;k<2;k++) {
                CVX_Voxel* v = Vx.setVoxel(pMaterial, i,j,k+1);
            }
        }
    }
    TI_VoxelyzeKernel VxKernel(&Vx);
    for (int j=0;j<steps;j++) {
        VxKernel.doTimeStep(0.00001);
        bool ret = Vx.doTimeStep(0.00001);
        if (!ret) {debugHost( printf("ERROR: Vx doTimeStep return false!") );break;}
        
        if (j%100==0) {
            debugHostx("step", printf("%d", j));
            VxKernel.readVoxelsPosFromDev();
            for (unsigned i=0;i<3;i++) {
                TI_Voxel* temp = VxKernel.read_voxels[i];
                debugDev( printf("[%d] Dev Position: %e, %e, %e", i, temp->pos.x, temp->pos.y, temp->pos.z) );
        
            }
            for (unsigned i=0;i<3;i++) {
                CVX_Voxel* temp = Vx.voxel(i);
                debugHost( printf("[%d] Dev Position: %e, %e, %e", i, temp->pos.x, temp->pos.y, temp->pos.z) );
            }
        }
    }
    printf("\n\n");
}