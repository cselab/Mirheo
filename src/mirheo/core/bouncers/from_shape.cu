// Copyright 2020 ETH Zurich. All Rights Reserved.
#include "from_shape.h"
#include "drivers/shape.h"
#include "kernels/api.h"

#include <mirheo/core/analytical_shapes/api.h>
#include <mirheo/core/celllist.h>
#include <mirheo/core/pvs/rigid_ashape_object_vector.h>
#include <mirheo/core/pvs/views/rsov.h>
#include <mirheo/core/rigid/operations.h>
#include <mirheo/core/utils/kernel_launch.h>

namespace mirheo
{

template <class Shape>
BounceFromRigidShape<Shape>::BounceFromRigidShape(const MirState *state,
                                                  const std::string& name,
                                                  VarBounceKernel varBounceKernel) :
    Bouncer(state, name),
    varBounceKernel_(varBounceKernel)
{}

template <class Shape>
BounceFromRigidShape<Shape>::~BounceFromRigidShape() = default;

template <class Shape>
void BounceFromRigidShape<Shape>::setup(ObjectVector *ov)
{
    Bouncer::setup(ov);

    ov->requireDataPerObject<RigidMotion> (channel_names::oldMotions, DataManager::PersistenceMode::Active, DataManager::ShiftMode::Active);
}

template <class Shape>
void BounceFromRigidShape<Shape>::setPrerequisites(ParticleVector *pv)
{
    // do not set it to persistent because bounce happens after integration
    pv->requireDataPerParticle<real4> (channel_names::oldPositions, DataManager::PersistenceMode::None, DataManager::ShiftMode::Active);
}

template <class Shape>
std::vector<std::string> BounceFromRigidShape<Shape>::getChannelsToBeExchanged() const
{
    return {channel_names::motions, channel_names::oldMotions};
}

template <class Shape>
std::vector<std::string> BounceFromRigidShape<Shape>::getChannelsToBeSentBack() const
{
    return {channel_names::motions}; // return forces and torque from remote bounce
}

template <class Shape>
void BounceFromRigidShape<Shape>::exec(ParticleVector *pv, CellList *cl, ParticleVectorLocality locality, cudaStream_t stream)
{
    auto rsov = dynamic_cast<RigidShapedObjectVector<Shape>*>(ov_);
    if (rsov == nullptr)
        die("Analytic %s bounce only works with Rigid %s vector", Shape::desc, Shape::desc);

    auto lrsov = rsov->get(locality);

    debug("Bouncing %d '%s' particles from %d '%s': objects (%s)",
          pv->local()->size(), pv->getCName(),
          lrsov->getNumObjects(), rsov->getCName(),
          getParticleVectorLocalityStr(locality).c_str());

    ov_->findExtentAndCOM(stream, locality);

    RSOVviewWithOldMotion<Shape> ovView(rsov, lrsov);
    PVviewWithOldParticles pvView(pv, pv->local());

    mpark::visit([&](auto& bounceKernel)
    {
        constexpr int nthreads = 256;
        const int nblocks = ovView.nObjects;
        const size_t smem = 2 * nthreads * sizeof(int);

        bounceKernel.update(rng_);

        SAFE_KERNEL_LAUNCH(
            shape_bounce_kernels::bounce,
            nblocks, nthreads, smem, stream,
            ovView, pvView, cl->cellInfo(), getState()->getDt(),
            bounceKernel);

    }, varBounceKernel_);
}

#define INSTANTIATE(Shape) template class BounceFromRigidShape<Shape>;

ASHAPE_TABLE(INSTANTIATE)

} // namespace mirheo
