#include "from_mesh.h"

#include <core/utils/kernel_launch.h>
#include <core/celllist.h>
#include <core/pvs/particle_vector.h>
#include <core/pvs/object_vector.h>

#include <core/rbc_kernels/bounce.h>
#include <core/cub/device/device_radix_sort.cuh>

/**
 * Create the bouncer
 * @param name unique bouncer name
 * @param kbT temperature which will be used to create a particle
 * velocity after the bounce, @see performBouncing()
 */
BounceFromMesh::BounceFromMesh(std::string name, float kbT) :
	Bouncer(name), nCollisions(1), kbT(kbT)
{	}


/**
 * @param ov will need an 'old_particles' per PARTICLE channel keeping positions
 * from the previous timestep.
 * This channel has to be communicated with the objects
 */
void BounceFromMesh::setup(ObjectVector* ov)
{
	this->ov = ov;

	// old positions HAVE to be known when the mesh travels to other ranks
	// shift HAS be applied as well
	ov->requireDataPerParticle<Particle> ("old_particles", true, sizeof(float));
}

/**
 * @brief Bounce particles from objects with meshes
 *
 * Firstly find all the collisions and generate array of colliding pairs Pid <--> TRid
 * Work is per-triangle, so only particle cell-lists are needed
 *
 * Secondly sort the array with respect to Pid
 *
 * Lastly resolve the collisions, choosing the first one in time for the same Pid
 */
void BounceFromMesh::exec(ParticleVector* pv, CellList* cl, float dt, bool local, cudaStream_t stream)
{
	auto activeOV = local ? ov->local() : ov->halo();

	debug("Bouncing %d '%s' particles from %d '%s' objects (%s)",
			pv->local()->size(), pv->name.c_str(),
			activeOV->nObjects,  ov->name.c_str(),
			local ? "local" : "halo");

	ov->findExtentAndCOM(stream, local);

	int totalTriangles = ov->mesh.ntriangles * activeOV->nObjects;

	// Set maximum possible number of collisions to bouncesPerTri * # of triangles
	// Generally, this is a vast overestimation of the actual number of collisions.
	// So if we've found more collisions something is surely broken
	nCollisions.clear(stream);
	collisionTable.resize_anew(bouncesPerTri*totalTriangles);
	tmp_collisionTable.resize_anew(bouncesPerTri*totalTriangles);

	int nthreads = 128;

	OVviewWithOldPartilces objView(ov, activeOV);
	PVviewWithOldParticles pvView(pv, pv->local());
	MeshView mesh(ov->mesh, activeOV->getMeshVertices(stream));

	// Step 1, find all the collistions
	SAFE_KERNEL_LAUNCH(
			findBouncesInMesh,
			getNblocks(totalTriangles, nthreads), nthreads, 0, stream,
			objView, pvView, mesh, cl->cellInfo(),
			nCollisions.devPtr(), collisionTable.devPtr(), collisionTable.size() );

	nCollisions.downloadFromDevice(stream);
	debug("Found %d collisions", nCollisions[0]);

	if (nCollisions[0] > collisionTable.size())
		die("Found too many collisions, something is likely broken");

	// Step 2, sort the collision table
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

	// Step 3, resolve the collisions
	SAFE_KERNEL_LAUNCH(
			performBouncing,
			getNblocks(nCollisions[0], nthreads), nthreads, 0, stream,
			objView, pvView, mesh,
			nCollisions[0], tmp_collisionTable.devPtr(), dt,
			kbT, drand48(), drand48() );
}
