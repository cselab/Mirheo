#include "from_shape.h"
#include "drivers/shape.h"
#include "kernels/api.h"

#include <mirheo/core/analytical_shapes/api.h>
#include <mirheo/core/celllist.h>
#include <mirheo/core/pvs/rigid_ashape_object_vector.h>
#include <mirheo/core/pvs/views/rsov.h>
#include <mirheo/core/rigid/operations.h>
#include <mirheo/core/utils/kernel_launch.h>

template <class Shape>
BounceFromRigidShape<Shape>::BounceFromRigidShape(const MirState *state,
                                                  const std::string& name,
                                                  VarBounceKernel varBounceKernel) :
    Bouncer(state, name),
    varBounceKernel(varBounceKernel)
{}

template <class Shape>
BounceFromRigidShape<Shape>::~BounceFromRigidShape() = default;

template <class Shape>
void BounceFromRigidShape<Shape>::setup(ObjectVector *ov)
{
    Bouncer::setup(ov);

    ov->requireDataPerObject<RigidMotion> (ChannelNames::oldMotions, DataManager::PersistenceMode::Active, DataManager::ShiftMode::Active);
}

template <class Shape>
void BounceFromRigidShape<Shape>::setPrerequisites(ParticleVector *pv)
{
    // do not set it to persistent because bounce happens after integration
    pv->requireDataPerParticle<real4> (ChannelNames::oldPositions, DataManager::PersistenceMode::None, DataManager::ShiftMode::Active);
}

template <class Shape>
std::vector<std::string> BounceFromRigidShape<Shape>::getChannelsToBeExchanged() const
{
    return {ChannelNames::motions, ChannelNames::oldMotions};
}

template <class Shape>
std::vector<std::string> BounceFromRigidShape<Shape>::getChannelsToBeSentBack() const
{
    return {ChannelNames::motions}; // return forces and torque from remote bounce
}

template <class Shape>
void BounceFromRigidShape<Shape>::exec(ParticleVector *pv, CellList *cl, ParticleVectorLocality locality, cudaStream_t stream)
{
    auto rsov = dynamic_cast<RigidShapedObjectVector<Shape>*>(ov);
    if (rsov == nullptr)
        die("Analytic %s bounce only works with Rigid %s vector", Shape::desc, Shape::desc);

    auto lrsov = rsov->get(locality);

    debug("Bouncing %d '%s' particles from %d '%s': objects (%s)",
          pv->local()->size(), pv->name.c_str(),
          lrsov->nObjects, rsov->name.c_str(),
          getParticleVectorLocalityStr(locality).c_str());

    ov->findExtentAndCOM(stream, locality);

    RSOVviewWithOldMotion<Shape> ovView(rsov, lrsov);
    PVviewWithOldParticles pvView(pv, pv->local());

    mpark::visit([&](auto& bounceKernel)
    {
        constexpr int nthreads = 256;
        const int nblocks = ovView.nObjects;
        const size_t smem = 2 * nthreads * sizeof(int);

        bounceKernel.update(rng);
    
        SAFE_KERNEL_LAUNCH(
            ShapeBounceKernels::bounce,
            nblocks, nthreads, smem, stream,
            ovView, pvView, cl->cellInfo(), state->dt,
            bounceKernel);
        
    }, varBounceKernel);
}

#define INSTANTIATE(Shape) template class BounceFromRigidShape<Shape>;

ASHAPE_TABLE(INSTANTIATE)

