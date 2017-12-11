#include "rigid_vv.h"

#include <core/utils/kernel_launch.h>
#include <core/logger.h>
#include <core/pvs/particle_vector.h>
#include <core/pvs/object_vector.h>
#include <core/pvs/rigid_object_vector.h>

#include <core/rigid_kernels/integration.h>


/**
 * Can only be applied to RigidObjectVector and requires it to have
 * \c old_motions data channel per particle
 */
void IntegratorVVRigid::setPrerequisites(ParticleVector* pv)
{
	auto ov = dynamic_cast<RigidObjectVector*> (pv);
	if (ov == nullptr)
		die("Rigid integration only works with rigid objects, can't work with %s", pv->name.c_str());

	ov->requireDataPerObject<RigidMotion>("old_motions", false);
	warn("Only objects with diagonal inertia tensors are supported now for rigid integration");
}


// FIXME: split VV into two stages
void IntegratorVVRigid::stage1(ParticleVector* pv, float t, cudaStream_t stream)
{}


/**
 * The function steps are as follows:
 *
 * - Collect the forces from the particles to get total force and torque per object
 * - Integrate object's COM coordinate RigidMotion::r and quatertion orientation
 *   RigidMotion::q. Velocity-Verlet is used for both.
 * - Rotate and translate the objects. For higher precision we don't use incremental
 *   updates, but rather take the initial particle coordinates
 *   RigidObjectVector::initialPositions and perform the full transformation for each
 *   object.
 * - Clear RigidMotion::force and RigidMotion::torque for each object.
 */
void IntegratorVVRigid::stage2(ParticleVector* pv, float t, cudaStream_t stream)
{
	auto ov = dynamic_cast<RigidObjectVector*> (pv);

	debug("Integrating %d rigid objects %s (total %d particles), timestep is %f",
			ov->local()->nObjects, ov->name.c_str(), ov->local()->size(), dt);

	ROVviewWithOldMotion ovView(ov, ov->local());

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
	pv->cellListStamp++;

	// Extents are changed too
	ov->local()->comExtentValid = false;
}

