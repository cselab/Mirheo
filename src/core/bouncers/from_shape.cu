#include "from_shape.h"
#include "drivers/shape.h"
#include "kernels/api.h"

#include <core/analytical_shapes/api.h>
#include <core/celllist.h>
#include <core/pvs/rigid_ashape_object_vector.h>
#include <core/pvs/views/rsov.h>
#include <core/rigid/operations.h>
#include <core/utils/kernel_launch.h>

template <class Shape>
BounceFromRigidShape<Shape>::BounceFromRigidShape(const MirState *state, std::string name) :
    Bouncer(state, name),
    varBounceKernel(BounceBack{})
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
    pv->requireDataPerParticle<float4> (ChannelNames::oldPositions, DataManager::PersistenceMode::None, DataManager::ShiftMode::Active);
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
void BounceFromRigidShape<Shape>::exec(ParticleVector *pv, CellList *cl, bool local, cudaStream_t stream)
{
    auto rsov = dynamic_cast<RigidShapedObjectVector<Shape>*>(ov);
    if (rsov == nullptr)
        die("Analytic %s bounce only works with Rigid %s vector", Shape::desc, Shape::desc);

    debug("Bouncing %d '%s' particles from %d '%s': objects (%s)",
          pv->local()->size(), pv->name.c_str(),
          local ? rsov->local()->nObjects : rsov->halo()->nObjects, rsov->name.c_str(),
          local ? "local objs" : "halo objs");

    ov->findExtentAndCOM(stream, local ? ParticleVectorType::Local : ParticleVectorType::Halo);

    RSOVviewWithOldMotion<Shape> ovView(rsov, local ? rsov->local() : rsov->halo());
    PVviewWithOldParticles pvView(pv, pv->local());

    if (!local)
        RigidOperations::clearRigidForcesFromMotions(ovView, stream);

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

