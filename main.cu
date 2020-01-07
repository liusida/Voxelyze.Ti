// #include <thrust/device_vector.h>
// #include "cuda.h"
// #include "cuda_runtime.h"

#include "Voxelyze.h"
#include "TI_VoxelyzeKernel.h"

int main() {
    CVoxelyze Vx(0.005); //5mm voxels
    Vx.enableFloor();
    Vx.enableCollisions();
    Vx.setGravity();
    CVX_Material* pMaterial = Vx.addMaterial(1000000, 1000); //A material with stiffness E=1MPa and density 1000Kg/m^3
    for (int i=0;i<3;i++) {
        for (int j=0;j<3;j++) {
            for (int k=0;k<3;k++) {
                CVX_Voxel* v = Vx.setVoxel(pMaterial, i,j,k+10);
            }
        }
    }
    Vx.voxel(0)->external()->setForce(Vec3D<float>(1,0,0));
    TI_VoxelyzeKernel VxKernel(&Vx);

}