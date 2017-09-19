#include "rigid_vv.h"

#include <core/logger.h>
#include <core/pvs/particle_vector.h>
#include <core/pvs/object_vector.h>
#include <core/pvs/rigid_object_vector.h>


/**
 * Assume that the forces are not yet distributed
 * Also integrate object's Q
 * VV integration now
 */
// FIXME: split VV into two stages
void IntegratorVVRigid::stage1(ParticleVector* pv, cudaStream_t stream)
{}

void IntegratorVVRigid::stage2(ParticleVector* pv, cudaStream_t stream)
{
	auto ov = dynamic_cast<RigidObjectVector*> (pv);
	if (ov == nullptr) die("Rigid integration only works with rigid objects, can't work with %s", pv->name.c_str());
	debug("Integrating %d rigid objects %s (total %d particles), timestep is %f", ov->local()->nObjects, ov->name.c_str(), ov->local()->size(), dt);
	if (ov->local()->nObjects == 0)
		return;

	warn("Only objects with diagonal inertia tensors are supported now for rigid integration");

	collectRigidForces<<< getNblocks(2*ov->local()->size(), 128), 128, 0, stream >>> (
			(float4*)ov->local()->coosvels.devPtr(), (float4*)ov->local()->forces.devPtr(), ov->local()->motions.devPtr(),
			ov->local()->comAndExtents.devPtr(), ov->local()->nObjects, ov->local()->objSize);

	const float3 J = ov->getInertiaTensor();
	const float3 J_1 = 1.0 / J;

	integrateRigidMotion<<< getNblocks(ov->local()->nObjects, 64), 64, 0, stream >>>(ov->local()->motions.devPtr(), J, J_1, 1.0 / ov->objMass, ov->local()->nObjects, dt);

	applyRigidMotion<<< getNblocks(ov->local()->size(), 128), 128, 0, stream >>>(
			(float4*)ov->local()->coosvels.devPtr(), ov->initialPositions.devPtr(), ov->local()->motions.devPtr(), ov->local()->nObjects, ov->objSize);

	clearRigidForces<<< getNblocks(ov->local()->nObjects, 64), 64, 0, stream >>>(ov->local()->motions.devPtr(), ov->local()->nObjects);

	pv->local()->changedStamp++;
}

