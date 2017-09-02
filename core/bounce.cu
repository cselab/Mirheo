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

#include <core/rbc_kernels/bounce.h>
#include <core/cub/device/device_radix_sort.cuh>

void BounceFromRigidEllipsoid::setup(ObjectVector* ov, ParticleVector* pv, CellList* cl)
{
	Bounce::setup(ov, pv, cl);
	rov = dynamic_cast<RigidObjectVector*>(ov);
	if (rov == nullptr)
		die("Analytic ellispoid bounce only works with RigidObjectVector");
}

void BounceFromRigidEllipsoid::exec(bool local, cudaStream_t stream)
{
	debug("Bouncing %s particles from %s objects", pv->name.c_str(), rov->name.c_str());
	auto activeROV = local ? rov->local() : rov->halo();

	int nthreads = 512;
	bounceEllipsoid<<< activeROV->nObjects, nthreads, 2*nthreads*sizeof(int), stream >>> (
			(float4*)pv->local()->coosvels.devPtr(), pv->mass, activeROV->comAndExtents.devPtr(), activeROV->motions.devPtr(),
			activeROV->nObjects, 1.0f / rov->axes, rov->axes,
			cl->cellsStartSize.devPtr(), cl->cellInfo(), dt);
}



/**
 * Firstly find all the collisions and generate array of colliding pairs Pid <--> TRid
 * Work is per-triangle, so only particle cell-lists are needed
 *
 * Secondly sort the array WRT Pid
 *
 * Lastly resolve the collisions, choosing the first one in time for the same Pid
 */
void BounceFromMesh::exec(bool local, cudaStream_t stream)
{
	debug("Bouncing %s particles from %s objects", pv->name.c_str(), ov->name.c_str());
	auto activeOV = local ? ov->local() : ov->halo();

	int totalTriangles = ov->mesh.ntriangles * activeOV->nObjects;

	nCollisions.clear(0);
	collisionTable.resize(bouncePerTri*totalTriangles, 0);
	tmp_collisionTable.resize(bouncePerTri*totalTriangles, 0);

	int nthreads = 128;

	findBouncesInMesh<<<getNblocks(totalTriangles, nthreads), nthreads>>> (
			(const float4*)pv->local()->coosvels.devPtr(), cl->cellsStartSize.devPtr(),
			cl->cellInfo(), nCollisions.devPtr(), collisionTable.devPtr(),
			activeOV->nObjects, ov->mesh.nvertices, ov->mesh.ntriangles, ov->mesh.triangles.devPtr(),
			(const float4*)activeOV->coosvels.devPtr(), dt);

	nCollisions.downloadFromDevice(0);

	debug2("Found %d collisions", nCollisions[0]);

	size_t bufSize;
	// Query for buffer size
	cub::DeviceRadixSort::SortKeys(nullptr, bufSize,
			(int64_t*)collisionTable.devPtr(), (int64_t*)tmp_collisionTable.devPtr(),
			nCollisions[0], 0, 32);
	// Allocate temporary storage
	sortBuffer.resize(bufSize, 0);
	// Run sorting operation
	cub::DeviceRadixSort::SortKeys(sortBuffer.devPtr(), bufSize,
			(int64_t*)collisionTable.devPtr(), (int64_t*)tmp_collisionTable.devPtr(),
			nCollisions[0], 0, 32);

	performBouncing<<< getNblocks(nCollisions[0], nthreads), nthreads >>> (
			nCollisions[0], tmp_collisionTable.devPtr(),
			(float4*)pv->local()->coosvels.devPtr(), pv->mass,
			ov->mesh.nvertices, ov->mesh.ntriangles, ov->mesh.triangles.devPtr(),
			(const float4*)activeOV->coosvels.devPtr(), (float*)activeOV->forces.devPtr(), ov->mass,
			dt);

}
