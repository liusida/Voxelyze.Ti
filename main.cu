#include <iostream>
#include <vector>
#include <stdio.h>

#include "glGraphics.h"
#include "Voxelyze.h"
#include "TI_VoxelyzeKernel.h"

using namespace std;

int main(int argc, char** argv) {
	glGraphics g;
    g.init();
    
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
    for (int i=0;i<3;i++) {
        for (int j=0;j<4;j++) {
            for (int k=0;k<5;k++) {
                CVX_Voxel* v = Vx.setVoxel(pMaterial, i,j,k+1);
            }
        }
    }
    TI_VoxelyzeKernel VxKernel(&Vx);
    unsigned j=0;
	while(g.running()) {
        for (int i=0;i<50;i++) 
        {
            VxKernel.doTimeStep(0.00001);
        }
        // bool ret = Vx.doTimeStep(0.00001);
        // if (!ret) {debugHost( printf("ERROR: Vx doTimeStep return false!") );break;}
        
        VxKernel.readVoxelsPosFromDev();
        std::vector<float> vertices;
        for (auto v:VxKernel.read_voxels) {
        // for (auto v:*Vx.voxelList()) {
            vertices.push_back(v->pos.y*20);
            vertices.push_back(v->pos.z*20);
            vertices.push_back(0.0f);
        }

        g.draw(vertices);

        if (j++>1000) {
            j=0;
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
}