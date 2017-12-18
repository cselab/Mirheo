#include "from_mesh.h"

#include <core/utils/kernel_launch.h>
#include <core/celllist.h>
#include <core/pvs/particle_vector.h>
#include <core/pvs/object_vector.h>

#include <core/rbc_kernels/bounce.h>
#include <core/cub/device/device_radix_sort.cuh>

#include <core/rigid_kernels/integration.h>

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

	// If the object is rigid, we need to collect the forces into the RigidMotion
	rov = dynamic_cast<RigidObjectVector*> (ov);

	// for NON-rigid objects:
	//
	// old positions HAVE to be known when the mesh travels to other ranks
	// shift HAS be applied as well
	//
	// for Rigid:
	// old motions HAVE to be there and communicated and shifted
	if (rov == nullptr)
		ov->requireDataPerParticle<Particle> ("old_particles", true, sizeof(float));
	else
		ov->requireDataPerObject<RigidMotion> ("old_motions", true, sizeof(RigidReal));
}

/**
 * Bounce particles from objects with meshes
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

	// FIXME this is a hack
	if (rov)
	{
		if (local)
			rov->local()->getMeshForces(stream)->clear(stream);
		else
			rov->halo()-> getMeshForces(stream)->clear(stream);
	}


	OVviewWithNewOldVertices vertexView(ov, activeOV, stream);
	PVviewWithOldParticles pvView(pv, pv->local());

	// Step 1, find all the collistions
	SAFE_KERNEL_LAUNCH(
			findBouncesInMesh,
			getNblocks(totalTriangles, nthreads), nthreads, 0, stream,
			vertexView, pvView, ov->mesh, cl->cellInfo(),
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
			vertexView, pvView, ov->mesh,
			nCollisions[0], tmp_collisionTable.devPtr(), dt,
			kbT, drand48(), drand48() );


	if (rov != nullptr)
	{
		// make a fake view with vertices instead of particles
		ROVview view(rov, local ? rov->local() : rov->halo());
		view.objSize = ov->mesh.nvertices;
		view.size = view.nObjects * view.objSize;
		view.particles = vertexView.vertices;
		view.forces = vertexView.vertexForces;

		SAFE_KERNEL_LAUNCH(
				collectRigidForces,
				getNblocks(view.size, 128), 128, 0, stream,
				view );
	}
}
