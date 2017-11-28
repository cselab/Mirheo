#include "from_mesh.h"

#include <core/utils/kernel_launch.h>
#include <core/celllist.h>
#include <core/pvs/particle_vector.h>
#include <core/pvs/object_vector.h>

#include <core/rbc_kernels/bounce.h>
#include <core/cub/device/device_radix_sort.cuh>

//FIXME this is a hack
__global__ void backtrack(PVviewWithOldParticles view, float dt)
{
	int gid = threadIdx.x + blockIdx.x*blockDim.x;
	if (gid >= view.size) return;

	Particle p(view.particles, gid);
	p.r -= dt*p.u;

	p.write2Float4(view.old_particles, gid);
}

/**
 * Firstly find all the collisions and generate array of colliding pairs Pid <--> TRid
 * Work is per-triangle, so only particle cell-lists are needed
 *
 * Secondly sort the array with respect to Pid
 *
 * Lastly resolve the collisions, choosing the first one in time for the same Pid
 */
void BounceFromMesh::exec(ParticleVector* pv, CellList* cl, float dt, cudaStream_t stream, bool local)
{
	auto activeOV = local ? ov->local() : ov->halo();

	debug("Bouncing %d '%s' particles from %d '%s' objects (%s)",
			pv->local()->size(), pv->name.c_str(),
			activeOV->nObjects,  ov->name.c_str(),
			local ? "local" : "halo");

	ov->findExtentAndCOM(stream, local);

	int totalTriangles = ov->mesh.ntriangles * activeOV->nObjects;

	nCollisions.clear(stream);
	collisionTable.resize_anew(bouncePerTri*totalTriangles);
	tmp_collisionTable.resize_anew(bouncePerTri*totalTriangles);

	int nthreads = 128;

	// FIXME do the hack

	if (!local)
	{
		return;
		PVviewWithOldParticles oldview(ov, activeOV);
		SAFE_KERNEL_LAUNCH(
				backtrack,
				getNblocks(totalTriangles, nthreads), nthreads, 0, stream,
				oldview, dt );
	}


	OVviewWithOldPartilces objView(ov, activeOV);
	PVviewWithOldParticles pvView(pv, pv->local());
	MeshView mesh(ov->mesh, activeOV->getMeshVertices(stream));

	SAFE_KERNEL_LAUNCH(
			findBouncesInMesh,
			getNblocks(totalTriangles, nthreads), nthreads, 0, stream,
			objView, pvView, mesh, cl->cellInfo(),
			nCollisions.devPtr(), collisionTable.devPtr(), collisionTable.size() );

	nCollisions.downloadFromDevice(stream);
	debug("Found %d collisions", nCollisions[0]);

	if (nCollisions[0] > collisionTable.size())
		die("Found too many collisions, something is likely broken");

	size_t bufSize;
	// Query for buffer size
	cub::DeviceRadixSort::SortKeys(nullptr, bufSize,
			(int64_t*)collisionTable.devPtr(), (int64_t*)tmp_collisionTable.devPtr(), nCollisions[0],
			0, 32, stream);
	// Allocate temporary storage
	sortBuffer.resize_anew(bufSize);
	// Run sorting operation
	cub::DeviceRadixSort::SortKeys(sortBuffer.devPtr(), bufSize,
			(int64_t*)collisionTable.devPtr(), (int64_t*)tmp_collisionTable.devPtr(), nCollisions[0],
			0, 32, stream);

	SAFE_KERNEL_LAUNCH(
			performBouncing,
			getNblocks(nCollisions[0], nthreads), nthreads, 0, stream,
			objView, pvView, mesh,
			nCollisions[0], tmp_collisionTable.devPtr(), dt,
			kbT, drand48(), drand48() );
}
