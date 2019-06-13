#include "from_rod.h"
#include "kernels/rod.h"

#include <core/celllist.h>
#include <core/pvs/rod_vector.h>
#include <core/pvs/views/rv.h>
#include <core/utils/kernel_launch.h>

BounceFromRod::BounceFromRod(const YmrState *state, std::string name, float radius, float kbT) :
    Bouncer(state, name),
    radius(radius),
    kbT(kbT)
{}

BounceFromRod::~BounceFromRod() = default;

void BounceFromRod::setup(ObjectVector *ov)
{
    Bouncer::setup(ov);

    rv = dynamic_cast<RodVector*> (ov);

    if (rv == nullptr)
        die("bounce from rod must be used with a rod vector");

    ov->requireDataPerParticle<float4> (ChannelNames::oldPositions, DataManager::PersistenceMode::Persistent, sizeof(float));
}

std::vector<std::string> BounceFromRod::getChannelsToBeExchanged() const
{
    return {ChannelNames::oldPositions};
}

void BounceFromRod::exec(ParticleVector *pv, CellList *cl, bool local, cudaStream_t stream)
{
    // auto activeRV = local ? rv->local() : rv->halo();

    // debug("Bouncing %d '%s' particles from %d '%s' rods (%s)",
    //       pv->local()->size(), pv->name.c_str(),
    //       activeRV->nObjects,  rv->name.c_str(),
    //       local ? "local" : "halo");

    // rv->findExtentAndCOM(stream, local ? ParticleVectorType::Local : ParticleVectorType::Halo);

    // int totalSegments = activeRV->getNumSegmentsPerRod() * activeRV->nObjects;

    // // Set maximum possible number of _coarse_ and _fine_ collisions with segments
    // // In case of crash, the estimate should be increased
    // int maxCoarseCollisions = coarseCollisionsPerSeg * totalSegments;
    // coarseTable.collisionTable.resize_anew(maxCoarseCollisions);
    // coarseTable.nCollisions.clear(stream);
    // MeshBounceKernels::TriangleTable devCoarseTable { maxCoarseCollisions,
    //                                                   coarseTable.nCollisions.devPtr(),
    //                                                   coarseTable.collisionTable.devPtr() };

    // int maxFineCollisions = fineCollisionsPerSeg * totalSegments;
    // fineTable.collisionTable.resize_anew(maxFineCollisions);
    // fineTable.nCollisions.clear(stream);
    // MeshBounceKernels::TriangleTable devFineTable { maxFineCollisions,
    //                                                 fineTable.nCollisions.devPtr(),
    //                                                 fineTable.collisionTable.devPtr() };

    // // Setup collision times array. For speed and simplicity initial time will be 0,
    // // and after the collisions detected its i-th element will be t_i-1.0f, where 0 <= t_i <= 1
    // // is the collision time, or 0 if no collision with the particle found
    // collisionTimes.resize_anew(pv->local()->size());
    // collisionTimes.clear(stream);

    // const int nthreads = 128;

    // activeRV->forces()->clear(stream);

    // OVviewWithNewOldVertices vertexView(ov, activeOV, stream);
    // PVviewWithOldParticles pvView(pv, pv->local());

    // // Step 1, find all the candidate collisions
    // SAFE_KERNEL_LAUNCH(
    //         BounceKernels::findBouncesInMesh,
    //         getNblocks(totalTriangles, nthreads), nthreads, 0, stream,
    //         vertexView, pvView, ov->mesh.get(), cl->cellInfo(), devCoarseTable );

    // coarseTable.nCollisions.downloadFromDevice(stream);
    // debug("Found %d triangle collision candidates", coarseTable.nCollisions[0]);

    // if (coarseTable.nCollisions[0] > maxCoarseCollisions)
    //     die("Found too many triangle collision candidates (%d),"
    //         "something may be broken or you need to increase the estimate", coarseTable.nCollisions[0]);

    // // Step 2, filter the candidates
    // SAFE_KERNEL_LAUNCH(
    //         BounceKernels::refineCollisions,
    //         getNblocks(coarseTable.nCollisions[0], nthreads), nthreads, 0, stream,
    //         vertexView, pvView, ov->mesh.get(),
    //         coarseTable.nCollisions[0], devCoarseTable.indices,
    //         devFineTable, collisionTimes.devPtr() );

    // fineTable.nCollisions.downloadFromDevice(stream);
    // debug("Found %d precise triangle collisions", fineTable.nCollisions[0]);

    // if (fineTable.nCollisions[0] > maxFineCollisions)
    //     die("Found too many precise triangle collisions (%d),"
    //         "something may be broken or you need to increase the estimate", fineTable.nCollisions[0]);


    // // Step 3, resolve the collisions
    // SAFE_KERNEL_LAUNCH(
    //         BounceKernels::performBouncingTriangle,
    //         getNblocks(fineTable.nCollisions[0], nthreads), nthreads, 0, stream,
    //         vertexView, pvView, ov->mesh.get(),
    //         fineTable.nCollisions[0], devFineTable.indices, collisionTimes.devPtr(),
    //         state->dt, kbT, drand48(), drand48() );

    // if (rov != nullptr)
    // {
    //     // make a fake view with vertices instead of particles
    //     ROVview view(rov, local ? rov->local() : rov->halo());
    //     view.objSize   = ov->mesh->getNvertices();
    //     view.size      = view.nObjects * view.objSize;
    //     view.positions = vertexView.vertices;
    //     view.forces    = vertexView.vertexForces;

    //     SAFE_KERNEL_LAUNCH(
    //             RigidIntegrationKernels::collectRigidForces,
    //             getNblocks(view.size, nthreads), nthreads, 0, stream,
    //             view );
    // }
}
