#include "TI_Voxel.h"

TI_Voxel::TI_Voxel(CVX_Voxel* p) {
    _voxel = p;
    pos = p->pos;
}
