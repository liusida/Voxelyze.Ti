#if !defined(TI_VEC3D_H)
#define TI_VEC3D_H

template <typename T = double>
class TI_Vec3D {
public:
    TI_Vec3D():x(0), y(0), z(0) {};
    TI_Vec3D(const T dx, const T dy, const T dz) {x = dx; y = dy; z = dz;} //!< Constructor with specified individual values.
	TI_Vec3D(const TI_Vec3D& s) {x = s.x; y = s.y; z = s.z;} //!< Copy constructor.

/* data */
	T x; //!< The current X value.
	T y; //!< The current Y value.
	T z; //!< The current Z value.

}

#endif // TI_VEC3D_H
