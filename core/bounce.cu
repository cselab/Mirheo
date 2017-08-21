/*
 * bounce.cu
 *
 *  Created on: Jul 20, 2017
 *      Author: alexeedm
 */

#include <core/bounce.h>
#include <core/particle_vector.h>
#include <core/celllist.h>
#include <core/rigid_object_vector.h>
#include <core/rigid_kernels/bounce.h>


void bounceFromRigidEllipsoid(ParticleVector* pv, CellList* cl, RigidObjectVector* rov, const float dt, bool local, cudaStream_t stream)
{
	debug("Bouncing %s particles from %s objects\n", pv->name.c_str(), rov->name.c_str());
	auto activeROV = local ? rov->local() : rov->halo();

	int nthreads = 512;
	bounceEllipsoid<<< activeROV->nObjects, nthreads, 2*nthreads*sizeof(int), stream >>> (
			(float4*)pv->local()->coosvels.devPtr(), pv->mass, activeROV->comAndExtents.devPtr(), activeROV->motions.devPtr(),
			activeROV->nObjects, 1.0f / rov->axes, rov->axes,
			cl->cellsStartSize.devPtr(), cl->cellInfo(), dt);
}
