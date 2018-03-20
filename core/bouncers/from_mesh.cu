#include "from_mesh.h"

#include <core/utils/kernel_launch.h>
#include <core/celllist.h>
#include <core/pvs/particle_vector.h>
#include <core/pvs/object_vector.h>

#include <core/membrane_kernels/bounce.h>
#include <core/cub/device/device_radix_sort.cuh>

#include <core/rigid_kernels/integration.h>

/**
 * Create the bouncer
 * @param name unique bouncer name
 * @param kbT temperature which will be used to create a particle
 * velocity after the bounce, @see performBouncing()
 */
BounceFromMesh::BounceFromMesh(std::string name, float kbT) :
	Bouncer(name), kbT(kbT)
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
 */
void BounceFromMesh::exec(ParticleVector* pv, CellList* cl, float dt, bool local, cudaStream_t stream)
{
	auto activeOV = local ? ov->local() : ov->halo();

	debug("Bouncing %d '%s' particles from %d '%s' objects (%s)",
			pv->local()->size(), pv->name.c_str(),
			activeOV->nObjects,  ov->name.c_str(),
			local ? "local" : "halo");

	ov->findExtentAndCOM(stream, local);

	int totalTriangles = ov->mesh->ntriangles * activeOV->nObjects;
	int totalEdges = totalTriangles * 3 / 2;

	// Set maximum possible number of _coarse_ and _fine_ collisions with triangles
	// In case of crash, the estimate should be increased
	int maxCoarseCollisions = coarseCollisionsPerTri * totalTriangles;
	coarseTable.nCollisions.clear(stream);
	coarseTable.collisionTable.resize_anew(maxCoarseCollisions);
	TriangleTable devCoarseTable { maxCoarseCollisions, coarseTable.nCollisions.devPtr(), coarseTable.collisionTable.devPtr() };

	int maxFineCollisions = fineCollisionsPerTri * totalTriangles;
	fineTable.nCollisions.clear(stream);
	fineTable.collisionTable.resize_anew(maxFineCollisions);
	TriangleTable devFineTable { maxFineCollisions, fineTable.nCollisions.devPtr(), fineTable.collisionTable.devPtr() };

	// Setup collision times array. For speed and simplicity initial time will be 0,
	// and after the collisions detected its i-th element will be t_i-1.0f, where 0 <= t_i <= 1
	// is the collision time, or 0 if no collision with the particle found
	collisionTimes.resize_anew(pv->local()->size());
	collisionTimes.clear(stream);

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

	// Step 1, find all the candidate collisions
	SAFE_KERNEL_LAUNCH(
			findBouncesInMesh,
			getNblocks(totalTriangles, nthreads), nthreads, 0, stream,
			vertexView, pvView, ov->mesh.get(), cl->cellInfo(), devCoarseTable );

	coarseTable.nCollisions.downloadFromDevice(stream);
	debug("Found %d triangle collision candidates", coarseTable.nCollisions[0]);

	if (coarseTable.nCollisions[0] > maxCoarseCollisions)
		die("Found too many triangle collision candidates, something may be broken or you need to increase the estimate");

	// Step 2, filter the candidates
	SAFE_KERNEL_LAUNCH(
			refineCollisions,
			getNblocks(coarseTable.nCollisions[0], nthreads), nthreads, 0, stream,
			vertexView, pvView, ov->mesh.get(),
			coarseTable.nCollisions[0], devCoarseTable.indices,
			devFineTable, collisionTimes.devPtr() );

	fineTable.nCollisions.downloadFromDevice(stream);
	debug("Found %d precise triangle collisions", fineTable.nCollisions[0]);

	if (fineTable.nCollisions[0] > maxFineCollisions)
		die("Found too many precise triangle collisions, something may be broken or you need to increase the estimate");


	// Step 3, resolve the collisions
	SAFE_KERNEL_LAUNCH(
			performBouncingTriangle,
			getNblocks(fineTable.nCollisions[0], nthreads), nthreads, 0, stream,
			vertexView, pvView, ov->mesh.get(),
			fineTable.nCollisions[0], devFineTable.indices, collisionTimes.devPtr(),
			dt, kbT, drand48(), drand48() );

	if (rov != nullptr)
	{
		// make a fake view with vertices instead of particles
		ROVview view(rov, local ? rov->local() : rov->halo());
		view.objSize = ov->mesh->nvertices;
		view.size = view.nObjects * view.objSize;
		view.particles = vertexView.vertices;
		view.forces = vertexView.vertexForces;

		SAFE_KERNEL_LAUNCH(
				collectRigidForces,
				getNblocks(view.size, 128), 128, 0, stream,
				view );
	}
}
