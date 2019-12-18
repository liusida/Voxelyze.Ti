#if !defined(TI_MATERIAL_H)
#define TI_MATERIAL_H

class CTI_Material
{
public:
    CTI_Material(float youngsModulus=1e6f, float density=1e3f); //!< Default Constructor. @param[in] youngsModulus The Young's Modulus (stiffness) of this material in Pascals. @param[in] density The density of this material in Kg/m^3
};




#endif // TI_MATERIAL_H
