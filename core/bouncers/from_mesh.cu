#include "from_mesh.h"

#include <core/utils/kernel_launch.h>
#include <core/celllist.h>
#include <core/pvs/particle_vector.h>
#include <core/pvs/rigid_ellipsoid_object_vector.h>

#include <core/rbc_kernels/bounce.h>
#include <core/cub/device/device_radix_sort.cuh>

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
	debug("Bouncing %s particles from %s objects", pv->name.c_str(), ov->name.c_str());
	auto activeOV = local ? ov->local() : ov->halo();

	int totalTriangles = ov->mesh.ntriangles * activeOV->nObjects;

	nCollisions.clear(stream);
	collisionTable.resize_anew(bouncePerTri*totalTriangles);
	tmp_collisionTable.resize_anew(bouncePerTri*totalTriangles);

	int nthreads = 128;

	// TODO: ovview with mesh
	SAFE_KERNEL_LAUNCH(
			findBouncesInMesh,
			getNblocks(totalTriangles, nthreads), nthreads, 0, stream,
			(const float4*)pv->local()->coosvels.devPtr(),
			cl->cellInfo(), nCollisions.devPtr(), collisionTable.devPtr(),
			activeOV->nObjects, ov->mesh.nvertices, ov->mesh.ntriangles, ov->mesh.triangles.devPtr(),
			(const float4*)activeOV->coosvels.devPtr(), dt );

	nCollisions.downloadFromDevice(stream);

	debug("Found %d collisions", nCollisions[0]);

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
			nCollisions[0], tmp_collisionTable.devPtr(),
			(float4*)pv->local()->coosvels.devPtr(), pv->mass,
			ov->mesh.nvertices, ov->mesh.ntriangles, ov->mesh.triangles.devPtr(),
			(const float4*)activeOV->coosvels.devPtr(), (float*)activeOV->forces.devPtr(), ov->mass,
			dt );
}
