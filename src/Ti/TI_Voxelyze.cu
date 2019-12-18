#include <new>

#include "TI_Voxelyze.h"

CTI_Voxelyze::CTI_Voxelyze(double voxelSize) {

}

bool CTI_Voxelyze::doTimeStep(float dt) {
	return true;
}

CTI_Material* CTI_Voxelyze::addMaterial(float youngsModulus, float density) {
    try {
		return NULL;
	}
    catch (std::bad_alloc&) {return NULL;}
}

// CTI_Voxel* CTI_Voxelyze::setVoxel(CTI_Material* material, int xIndex, int yIndex, int zIndex)
// {
// 	// if (material == NULL){
// 	// 	removeVoxel(xIndex, yIndex, zIndex);
// 	// 	return NULL;
// 	// }
	
// 	CTI_Voxel* pV = voxels(xIndex, yIndex, zIndex);
// 	// if (pV != NULL){
// 	// 	replaceVoxel((CTI_MaterialVoxel*)material, xIndex, yIndex, zIndex);
// 	 	return pV;
// 	// }
// 	// else {
// 	// 	return addVoxel((CTI_MaterialVoxel*)material, xIndex, yIndex, zIndex);
// 	// }
// }

// CTI_Voxel* CTI_Voxelyze::voxels() {

// }
