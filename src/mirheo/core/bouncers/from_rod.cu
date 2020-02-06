#include "from_rod.h"
#include "drivers/rod.h"

#include <mirheo/core/celllist.h>
#include <mirheo/core/pvs/rod_vector.h>
#include <mirheo/core/pvs/views/rv.h>
#include <mirheo/core/utils/kernel_launch.h>

namespace mirheo
{

BounceFromRod::BounceFromRod(const MirState *state,
                             const std::string& name,
                             real radius,
                             VarBounceKernel varBounceKernel) :
    Bouncer(state, name),
    radius_(radius),
    varBounceKernel_(varBounceKernel)
{}

BounceFromRod::~BounceFromRod() = default;

void BounceFromRod::setup(ObjectVector *ov)
{
    Bouncer::setup(ov);

    rv_ = dynamic_cast<RodVector*> (ov);

    if (rv_ == nullptr)
        die("bounce from rod must be used with a rod vector");

    ov->requireDataPerParticle<real4> (ChannelNames::oldPositions, DataManager::PersistenceMode::Active, DataManager::ShiftMode::Active);
}

void BounceFromRod::setPrerequisites(ParticleVector *pv)
{
    // do not set it to persistent because bounce happens after integration
    pv->requireDataPerParticle<real4> (ChannelNames::oldPositions, DataManager::PersistenceMode::None, DataManager::ShiftMode::Active);
}

std::vector<std::string> BounceFromRod::getChannelsToBeExchanged() const
{
    return {ChannelNames::oldPositions};
}

std::vector<std::string> BounceFromRod::getChannelsToBeSentBack() const
{
    return {ChannelNames::forces};
}

void BounceFromRod::exec(ParticleVector *pv, CellList *cl, ParticleVectorLocality locality, cudaStream_t stream)
{
    auto activeRV = rv_->get(locality);

    debug("Bouncing %d '%s' particles from %d '%s' rods (%s)",
          pv->local()->size(), pv->getCName(),
          activeRV->getNumObjects(),  rv_->getCName(),
          getParticleVectorLocalityStr(locality).c_str());

    rv_->findExtentAndCOM(stream, locality);

    const int totalSegments = activeRV->getNumSegmentsPerRod() * activeRV->getNumObjects();

    // Set maximum possible number of collisions with segments
    // In case of crash, the estimate should be increased
    const int maxCollisions = static_cast<int>(collisionsPerSeg_ * static_cast<real>(totalSegments));
    table_.collisionTable.resize_anew(maxCollisions);
    table_.nCollisions.clear(stream);
    RodBounceKernels::SegmentTable devCollisionTable { maxCollisions,
                                                       table_.nCollisions.devPtr(),
                                                       table_.collisionTable.devPtr() };


    // Setup collision times array. For speed and simplicity initial time will be 0,
    // and after the collisions detected its i-th element will be t_i-1.0_r, where 0 <= t_i <= 1
    // is the collision time, or 0 if no collision with the particle found
    collisionTimes.resize_anew(pv->local()->size());
    collisionTimes.clear(stream);

    const int nthreads = 128;

    activeRV->forces().clear(stream);

    RVviewWithOldParticles rvView(rv_, activeRV);
    PVviewWithOldParticles pvView(pv, pv->local());

    // Step 1, find all the candidate collisions
    SAFE_KERNEL_LAUNCH(
            RodBounceKernels::findBounces,
            getNblocks(totalSegments, nthreads), nthreads, 0, stream,
            rvView, radius_, pvView, cl->cellInfo(), devCollisionTable, collisionTimes.devPtr() );

    table_.nCollisions.downloadFromDevice(stream);
    const int nCollisions = table_.nCollisions[0];
    debug("Found %d rod collision candidates", nCollisions);

    if (nCollisions > maxCollisions)
        die("Found too many rod collisions (%d),"
            "something may be broken or you need to increase the estimate", nCollisions);

    // Step 2, resolve the collisions
    mpark::visit([&](auto& bounceKernel)
    {
        bounceKernel.update(rng_);
    
        SAFE_KERNEL_LAUNCH(
            RodBounceKernels::performBouncing,
            getNblocks(nCollisions, nthreads), nthreads, 0, stream,
            rvView, radius_, pvView, nCollisions, devCollisionTable.indices, collisionTimes.devPtr(),
            getState()->dt, bounceKernel);

    }, varBounceKernel_);
}

} // namespace mirheo
