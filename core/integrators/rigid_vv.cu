#include "rigid_vv.h"

#include <core/utils/kernel_launch.h>
#include <core/logger.h>
#include <core/pvs/particle_vector.h>
#include <core/pvs/object_vector.h>
#include <core/pvs/rigid_object_vector.h>

#include <core/rigid_kernels/integration.h>


/**
 * Assume that the forces are not yet distributed
 * Also integrate object's Q
 * VV integration now
 */
// FIXME: split VV into two stages
void IntegratorVVRigid::stage1(ParticleVector* pv, float t, cudaStream_t stream)
{}

void IntegratorVVRigid::stage2(ParticleVector* pv, float t, cudaStream_t stream)
{
	auto ov = dynamic_cast<RigidObjectVector*> (pv);
	if (ov == nullptr)
		die("Rigid integration only works with rigid objects, can't work with %s", pv->name.c_str());

	debug("Integrating %d rigid objects %s (total %d particles), timestep is %f",
			ov->local()->nObjects, ov->name.c_str(), ov->local()->size(), dt);

	if (ov->local()->nObjects == 0)
		return;

	ROVview ovView(ov, ov->local());

	warn("Only objects with diagonal inertia tensors are supported now for rigid integration");

	SAFE_KERNEL_LAUNCH(
			collectRigidForces,
			getNblocks(2*ovView.size, 128), 128, 0, stream,
			ovView );

	SAFE_KERNEL_LAUNCH(
			integrateRigidMotion,
			getNblocks(ovView.nObjects, 64), 64, 0, stream,
			ovView, dt );

	SAFE_KERNEL_LAUNCH(
			applyRigidMotion,
			getNblocks(ovView.size, 128), 128, 0, stream,
			ovView, ov->initialPositions.devPtr() );

	SAFE_KERNEL_LAUNCH(
			clearRigidForces,
			getNblocks(ovView.nObjects, 64), 64, 0, stream,
			ovView );

	// PV may have changed, invalidate all
	pv->haloValid = false;
	pv->redistValid = false;
	pv->celllistValid = false;
}

