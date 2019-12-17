#include <iostream>

#include "Voxelyze.h"
#include "TI_Object.h"

int main(int, char**) {
    CVoxelyze Vx(0.005); //5mm voxels
    CVX_Material* pMaterial = Vx.addMaterial(1000000, 1000); //A material with stiffness E=1MPa and density 1000Kg/m^3
    CVX_Voxel* Voxel1 = Vx.setVoxel(pMaterial, 0, 0, 0); //Voxel at index x=0, y=0. z=0
    CVX_Voxel* Voxel2 = Vx.setVoxel(pMaterial, 1, 1, 0);
    CVX_Voxel* Voxel3 = Vx.setVoxel(pMaterial, 2, 0, 0); //Beam extends in the +X direction

    for (unsigned i=0;i<1e3;i++) {
        Vx.setVoxel(pMaterial, -1,-1, i);
    }

    Voxel1->external()->setFixedAll(); //Fixes all 6 degrees of freedom with an external condition on Voxel 1
    Voxel3->external()->setForce(0, 0, 1); //pulls Voxel 3 downward with 1 Newton of force.

    // for (int i=0; i<1e7; i++) {
    //     Vx.doTimeStep(0.001); //simulate  100 timesteps.
    //     if (i%1000==0)
    //         std::cout<< i<<") V1: "<<Voxel1->position().z<< " V2: "<<Voxel2->position().z<< " V3: " << Voxel3->position().z << std::endl;
    // }

    CTI_Object tmp;
    while(1)
        tmp.Try();

}
