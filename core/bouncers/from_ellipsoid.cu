/*
 * bounce.cu
 *
 *  Created on: Jul 20, 2017
 *      Author: alexeedm
 */

#include "from_ellipsoid.h"

#include <core/celllist.h>
#include <core/pvs/particle_vector.h>
#include <core/pvs/rigid_ellipsoid_object_vector.h>

#include <core/rigid_kernels/bounce.h>

void BounceFromRigidEllipsoid::exec(ObjectVector* ov, ParticleVector* pv, CellList* cl, float dt, cudaStream_t stream, bool local)
{
	auto reov = dynamic_cast<RigidEllipsoidObjectVector*>(ov);
	if (reov == nullptr)
		die("Analytic ellispoid bounce only works with RigidObjectVector");

	debug("Bouncing %s particles from %s object vector", pv->name.c_str(), reov->name.c_str());
	auto activeREOV = local ? reov->local() : reov->halo();

	int nthreads = 512;
	bounceEllipsoid<<< activeREOV->nObjects, nthreads, 2*nthreads*sizeof(int), stream >>> (
			(float4*)pv->local()->coosvels.devPtr(), pv->mass, activeREOV->comAndExtents.devPtr(), activeREOV->motions.devPtr(),
			activeREOV->nObjects, 1.0f / reov->axes, reov->axes,
			cl->cellsStartSize.devPtr(), cl->cellInfo(), dt);
}



