#include "from_mesh.h"
#include "drivers/mesh.h"
#include "kernels/api.h"

#include <mirheo/core/celllist.h>
#include <mirheo/core/pvs/object_vector.h>
#include <mirheo/core/pvs/rigid_object_vector.h>
#include <mirheo/core/pvs/views/ov.h>
#include <mirheo/core/rigid/operations.h>
#include <mirheo/core/utils/kernel_launch.h>

namespace mirheo
{

BounceFromMesh::BounceFromMesh(const MirState *state, const std::string& name, VarBounceKernel varBounceKernel) :
    Bouncer(state, name),
    varBounceKernel_(varBounceKernel)
{}

BounceFromMesh::~BounceFromMesh() = default;

void BounceFromMesh::setup(ObjectVector *ov)
{
    Bouncer::setup(ov);

    // If the object is rigid, we need to collect the forces into the RigidMotion
    rov_ = dynamic_cast<RigidObjectVector*> (ov);

    // for NON-rigid objects:
    //
    // old positions HAVE to be known when the mesh travels to other ranks
    // shift HAS be applied as well
    //
    // for Rigid:
    // old motions HAVE to be there and communicated and shifted

    if (rov_ == nullptr)
        ov->requireDataPerParticle<real4> (ChannelNames::oldPositions, DataManager::PersistenceMode::Active, DataManager::ShiftMode::Active);
    else
        ov->requireDataPerObject<RigidMotion> (ChannelNames::oldMotions, DataManager::PersistenceMode::Active, DataManager::ShiftMode::Active);
}

void BounceFromMesh::setPrerequisites(ParticleVector *pv)
{
    // do not set it to persistent because bounce happens after integration
    pv->requireDataPerParticle<real4> (ChannelNames::oldPositions, DataManager::PersistenceMode::None, DataManager::ShiftMode::Active);
}

std::vector<std::string> BounceFromMesh::getChannelsToBeExchanged() const
{
    if (rov_)
        return {ChannelNames::motions, ChannelNames::oldMotions};
    else
        return {ChannelNames::oldPositions};
}

std::vector<std::string> BounceFromMesh::getChannelsToBeSentBack() const
{
    if (rov_)
        return {ChannelNames::motions};
    else
        // return {ChannelNames::forces};
        return {};
}

void BounceFromMesh::exec(ParticleVector *pv, CellList *cl, ParticleVectorLocality locality, cudaStream_t stream)
{
    auto activeOV = ov_->get(locality);

    debug("Bouncing %d '%s' particles from %d '%s' objects (%s)",
          pv->local()->size(), pv->getCName(),
          activeOV->getNumObjects(),  ov_->getCName(),
          getParticleVectorLocalityStr(locality).c_str());

    ov_->findExtentAndCOM(stream, locality);

    const int totalTriangles = ov_->mesh->getNtriangles() * activeOV->getNumObjects();

    // Set maximum possible number of _coarse_ and _fine_ collisions with triangles
    // In case of crash, the estimate should be increased
    const int maxCoarseCollisions = static_cast<int>(coarseCollisionsPerTri_ * static_cast<real>(totalTriangles));
    coarseTable_.collisionTable.resize_anew(maxCoarseCollisions);
    coarseTable_.nCollisions.clear(stream);
    MeshBounceKernels::TriangleTable devCoarseTable { maxCoarseCollisions,
                                                      coarseTable_.nCollisions.devPtr(),
                                                      coarseTable_.collisionTable.devPtr() };
    
    const int maxFineCollisions = static_cast<int>(fineCollisionsPerTri_ * static_cast<real>(totalTriangles));
    fineTable_.collisionTable.resize_anew(maxFineCollisions);
    fineTable_.nCollisions.clear(stream);
    MeshBounceKernels::TriangleTable devFineTable { maxFineCollisions,
                                                    fineTable_.nCollisions.devPtr(),
                                                    fineTable_.collisionTable.devPtr() };

    // Setup collision times array. For speed and simplicity initial time will be 0,
    // and after the collisions detected its i-th element will be t_i-1.0_r, where 0 <= t_i <= 1
    // is the collision time, or 0 if no collision with the particle found
    collisionTimes_.resize_anew(pv->local()->size());
    collisionTimes_.clear(stream);

    const int nthreads = 128;

    // FIXME this is a hack
    if (rov_)
    {
        if (locality == ParticleVectorLocality::Local)
            rov_->local()->getMeshForces(stream)->clear(stream);
        else
            rov_->halo()-> getMeshForces(stream)->clear(stream);
    }


    OVviewWithNewOldVertices vertexView(ov_, activeOV, stream);
    PVviewWithOldParticles pvView(pv, pv->local());

    // Step 1, find all the candidate collisions
    SAFE_KERNEL_LAUNCH(
            MeshBounceKernels::findBouncesInMesh,
            getNblocks(totalTriangles, nthreads), nthreads, 0, stream,
            vertexView, pvView, ov_->mesh.get(), cl->cellInfo(), devCoarseTable );

    coarseTable_.nCollisions.downloadFromDevice(stream);
    debug("Found %d triangle collision candidates", coarseTable_.nCollisions[0]);

    if (coarseTable_.nCollisions[0] > maxCoarseCollisions)
        die("Found too many triangle collision candidates (coarse) (%d, max %d),"
            "something may be broken or you need to increase the estimate",
            coarseTable_.nCollisions[0], maxCoarseCollisions);

    // Step 2, filter the candidates
    SAFE_KERNEL_LAUNCH(
            MeshBounceKernels::refineCollisions,
            getNblocks(coarseTable_.nCollisions[0], nthreads), nthreads, 0, stream,
            vertexView, pvView, ov_->mesh.get(),
            coarseTable_.nCollisions[0], devCoarseTable.indices,
            devFineTable, collisionTimes_.devPtr() );

    fineTable_.nCollisions.downloadFromDevice(stream);
    debug("Found %d precise triangle collisions", fineTable_.nCollisions[0]);

    if (fineTable_.nCollisions[0] > maxFineCollisions)
        die("Found too many triangle collisions (precise) (%d, max %d),"
            "something may be broken or you need to increase the estimate",
            fineTable_.nCollisions[0], maxFineCollisions);

    // Step 3, resolve the collisions    
    mpark::visit([&](auto& bounceKernel)
    {
        bounceKernel.update(rng_);
        
        SAFE_KERNEL_LAUNCH(
            MeshBounceKernels::performBouncingTriangle,
            getNblocks(fineTable_.nCollisions[0], nthreads), nthreads, 0, stream,
            vertexView, pvView, ov_->mesh.get(),
            fineTable_.nCollisions[0], devFineTable.indices, collisionTimes_.devPtr(),
            getState()->dt, bounceKernel );

    }, varBounceKernel_);

    if (rov_)
    {
        // make a fake view with vertices instead of particles
        ROVview view(rov_, rov_->get(locality));
        view.objSize   = ov_->mesh->getNvertices();
        view.size      = view.nObjects * view.objSize;
        view.positions = vertexView.vertices;
        view.forces    = vertexView.vertexForces;

        RigidOperations::collectRigidForces(view, stream);
    }
}

} // namespace mirheo
