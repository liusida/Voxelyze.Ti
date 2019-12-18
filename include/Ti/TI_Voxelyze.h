#ifndef TI_VOXELYZE_H
#define TI_VOXELYZE_H

#include "TI_Material.h"
#include "TI_Voxel.h"

#define DEFAULT_VOXEL_SIZE 0.001 //1mm default voxel size

class CTI_Voxelyze {
public:
	CTI_Voxelyze(double voxelSize = DEFAULT_VOXEL_SIZE); //!< Constructs an empty voxelyze object. @param[in] voxelSize base size of the voxels in this instance in meters.
	bool doTimeStep(float dt = -1.0f); //!< Executes a single timestep on this voxelyze object and updates all state information (voxel positions and orientations) accordingly. In most situations this function will be called repeatedly until the desired result is obtained. @param[in] dt The timestep to take in seconds. If this value is too large the system will display divergent instability. Use recommendedTimeStep() to get a conservative estimate of the largest stable timestep. Also the default value of -1.0f will blindly use this recommended timestep.
	CTI_Material* addMaterial(float youngsModulus = 1e6f, float density = 1e3f); //!< Adds a material to this voxelyze object with the minimum necessary information for dynamic simulation (stiffness, density). Returns a pointer to the newly created material that can be used to further specify properties using CTI_Material public memer functions. See CTI_Material documentation. This function does not create any voxels, but a returned CTI_material pointer is a necessary parameter for the setVoxel() function that does add voxels. @param[in] youngsModulus the desired stiffness (Young's Modulus) of this material in Pa (N/m^2). @param[in] density the desired density of this material in Kg/m^3.
	CTI_Voxel* setVoxel(CTI_Material* material, int xIndex, int yIndex, int zIndex); //!< Adds a voxel made of material at the specified index. If a voxel already exists here it is replaced. The returned pointer can be safely modified by calling any CTI_Voxel public member function on it. @param[in] material material this voxel is made from. This material must already be a part of the simulation - the pointer will have originated from addMaterial() or material(). @param[in] xIndex the X index of this voxel. @param[in] yIndex the Y index of this voxel. @param[in] zIndex the Z index of this voxel.

	//CTI_Voxel* addVoxel(CTI_MaterialVoxel* newVoxelMaterial, int xIndex, int yIndex, int zIndex); //creates a new voxel if there isn't one here. Otherwise
	//void removeVoxel(int xIndex, int yIndex, int zIndex);
	//void replaceVoxel(CTI_MaterialVoxel* newVoxelMaterial, int xIndex, int yIndex, int zIndex); //replaces the material of this voxel while retaining its position, velocity, etc.

private:
    double voxSize; //lattice size
};

#endif //TI_VOXELYZE_H