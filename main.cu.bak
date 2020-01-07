#include <iostream>
#include <vector>
#include <stdio.h>

#include "glGraphics.h"
#include "Voxelyze.h"
#include "TI_VoxelyzeKernel.h"

#define GPU
// #define CPU

int main(int argc, char** argv) {
	glGraphics g;
    g.init();
    CVoxelyze Vx(0.005); //5mm voxels
    Vx.enableFloor();
    Vx.enableCollisions();
    Vx.setGravity();
    CVX_Material* pMaterial = Vx.addMaterial(1000000, 1000); //A material with stiffness E=1MPa and density 1000Kg/m^3
    for (int i=0;i<20;i++) {
        for (int j=0;j<20;j++) {
            for (int k=0;k<20;k++) {
                CVX_Voxel* v = Vx.setVoxel(pMaterial, i,j,k+10);
            }
        }
    }
    Vx.voxel(0)->external()->setForce(Vec3D<float>(1,0,0));

    // for (int i=0;i<7;i++) {
    //     for (int j=0;j<7;j++) {
    //         for (int k=0;k<7;k++) {
    //             CVX_Voxel* v = Vx.setVoxel(pMaterial, i,j-8,k+10);
    //         }
    //     }
    // }

    #ifdef GPU
    TI_VoxelyzeKernel VxKernel(&Vx);
    #endif

    #ifdef GPU
    printf("GPU enabled.\n");
    #endif
    
    #ifdef CPU
    printf("CPU enabled.\n");
    #endif

    double time_step = Vx.recommendedTimeStep();
    printf("time_step %f.\n", time_step);

    unsigned j=0;
	while(g.running()) {
        for (int i=0;i<200;i++) 
        {
            #ifdef GPU
            VxKernel.doTimeStep(time_step);
            #endif
            #ifdef CPU
            bool ret = Vx.doTimeStep(time_step);
            if (!ret) {debugHost( printf("ERROR: Vx doTimeStep return false!") );break;}
            #endif
        }
        #ifdef GPU
        VxKernel.readVoxelsPosFromDev();
        #endif

        g.clear();

        #ifdef GPU
        std::vector<float> vertices_gpu;
        for (auto v:VxKernel.read_voxels) {
            vertices_gpu.push_back(v->pos.y*10+0.35);
            vertices_gpu.push_back(v->pos.z*10);
            vertices_gpu.push_back(0.0f);
        }
        g.draw(vertices_gpu,1);
        #endif

        #ifdef CPU
        std::vector<float> vertices_cpu;
        for (auto v:*(Vx.voxelList())) {
            vertices_cpu.push_back(v->pos.y*10-0.35);
            vertices_cpu.push_back(v->pos.z*10);
            vertices_cpu.push_back(0.0f);
        }
        g.draw(vertices_cpu,0);
        #endif

        g.swap();

        if (j++>100) break;
    }
    #ifdef GPU
    for (unsigned i=0;i<3;i++) {
        TI_Voxel* temp = VxKernel.read_voxels[i];
        debugDev( printf("[%d] Dev Position: %e, %e, %e", i, temp->pos.x, temp->pos.y, temp->pos.z) );
    }
    #endif
    #ifdef CPU
    for (unsigned i=0;i<3;i++) {
        CVX_Voxel* temp = Vx.voxel(i);
        debugHost( printf("[%d] Host Position: %e, %e, %e", i, temp->pos.x, temp->pos.y, temp->pos.z) );
    }
    #endif
}
