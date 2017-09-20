#include "rigid_ellipsoid_object_vector.h"

#include <core/rigid_kernels/quaternion.h>
#include <core/cuda_common.h>

float3 RigidEllipsoidObjectVector::getInertiaTensor()
{
	return objMass / 5.0 * make_float3(
			sqr(axes.y) + sqr(axes.z),
			sqr(axes.z) + sqr(axes.x),
			sqr(axes.x) + sqr(axes.y) );
}




