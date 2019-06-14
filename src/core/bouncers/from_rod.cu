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

void BounceFromRod::setPrerequisites(ParticleVector *pv)
{
    // do not set it to persistent because bounce happens after integration
    pv->requireDataPerParticle<float4> (ChannelNames::oldPositions, DataManager::PersistenceMode::None, sizeof(float));
}

std::vector<std::string> BounceFromRod::getChannelsToBeExchanged() const
{
    return {ChannelNames::oldPositions};
}

void BounceFromRod::exec(ParticleVector *pv, CellList *cl, bool local, cudaStream_t stream)
{
    auto activeRV = local ? rv->local() : rv->halo();

    debug("Bouncing %d '%s' particles from %d '%s' rods (%s)",
          pv->local()->size(), pv->name.c_str(),
          activeRV->nObjects,  rv->name.c_str(),
          local ? "local" : "halo");

    rv->findExtentAndCOM(stream, local ? ParticleVectorType::Local : ParticleVectorType::Halo);

    int totalSegments = activeRV->getNumSegmentsPerRod() * activeRV->nObjects;

    // Set maximum possible number of collisions with segments
    // In case of crash, the estimate should be increased
    int maxCollisions = collisionsPerSeg * totalSegments;
    table.collisionTable.resize_anew(maxCollisions);
    table.nCollisions.clear(stream);
    RodBounceKernels::SegmentTable devCollisionTable { maxCollisions,
                                                       table.nCollisions.devPtr(),
                                                       table.collisionTable.devPtr() };


    // Setup collision times array. For speed and simplicity initial time will be 0,
    // and after the collisions detected its i-th element will be t_i-1.0f, where 0 <= t_i <= 1
    // is the collision time, or 0 if no collision with the particle found
    collisionTimes.resize_anew(pv->local()->size());
    collisionTimes.clear(stream);

    const int nthreads = 128;

    activeRV->forces().clear(stream);

    RVviewWithOldParticles rvView(rv, activeRV);
    PVviewWithOldParticles pvView(pv, pv->local());

    // Step 1, find all the candidate collisions
    SAFE_KERNEL_LAUNCH(
            RodBounceKernels::findBounces,
            getNblocks(totalSegments, nthreads), nthreads, 0, stream,
            rvView, radius, pvView, cl->cellInfo(), devCollisionTable, collisionTimes.devPtr() );

    table.nCollisions.downloadFromDevice(stream);
    int nCollisions = table.nCollisions[0];
    debug("Found %d rod collision candidates", nCollisions);

    if (table.nCollisions[0] > maxCollisions)
        die("Found too many rod collisions (%d),"
            "something may be broken or you need to increase the estimate", nCollisions);

    // Step 2, resolve the collisions
    SAFE_KERNEL_LAUNCH(
            RodBounceKernels::performBouncing,
            getNblocks(table.nCollisions[0], nthreads), nthreads, 0, stream,
            rvView, radius, pvView, nCollisions, devCollisionTable.indices, collisionTimes.devPtr(),
            state->dt, kbT, drand48(), drand48() );
}
