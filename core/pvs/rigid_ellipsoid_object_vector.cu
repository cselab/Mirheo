#include "rigid_ellipsoid_object_vector.h"

#include <core/rigid_kernels/quaternion.h>
#include <core/utils/cuda_common.h>

float3 RigidEllipsoidObjectVector::getInertiaTensor()
{
	return mass*objSize / 5.0 * make_float3(
			sqr(axes.y) + sqr(axes.z),
			sqr(axes.z) + sqr(axes.x),
			sqr(axes.x) + sqr(axes.y) );
}




