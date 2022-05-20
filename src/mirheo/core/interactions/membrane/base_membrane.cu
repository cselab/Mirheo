// Copyright 2020 ETH Zurich. All Rights Reserved.
#include "base_membrane.h"

#include "force_kernels/common.h"

#include <mirheo/core/pvs/membrane_vector.h>
#include <mirheo/core/pvs/views/ov.h>
#include <mirheo/core/utils/cuda_common.h>
#include <mirheo/core/utils/kernel_launch.h>
#include <mirheo/core/utils/macros.h>

namespace mirheo
{

namespace base_membrane_interaction_kernels
{
__global__ void computeAreaAndVolume(OVviewWithAreaVolume view, MeshView mesh)
{
    const int objId = blockIdx.x;
    const int offset = objId * mesh.nvertices;
    real2 a_v = make_real2(0.0_r);

    for (int i = threadIdx.x; i < mesh.ntriangles; i += blockDim.x) {
        const int3 ids = mesh.triangles[i];

        const auto v0 = make_mReal3(make_real3( view.readPosition(offset + ids.x) ));
        const auto v1 = make_mReal3(make_real3( view.readPosition(offset + ids.y) ));
        const auto v2 = make_mReal3(make_real3( view.readPosition(offset + ids.z) ));

        a_v.x += triangleArea(v0, v1, v2);
        a_v.y += triangleSignedVolume(v0, v1, v2);
    }

    a_v = warpReduce( a_v, [] (real a, real b) { return a+b; } );

    if (laneId() == 0)
        atomicAdd(&view.area_volumes[objId], a_v);
}
} // namespace base_membrane_interaction_kernels


BaseMembraneInteraction::BaseMembraneInteraction(const MirState *state, const std::string& name) :
    Interaction(state, name)
{}

BaseMembraneInteraction::~BaseMembraneInteraction() = default;


void BaseMembraneInteraction::setPrerequisites(ParticleVector *pv1, ParticleVector *pv2,
                                               __UNUSED  CellList *cl1, __UNUSED  CellList *cl2)
{
    if (pv1 != pv2)
        die("Internal membrane forces can't be computed between two different particle vectors");

    if (auto mv = dynamic_cast<MembraneVector*>(pv1))
    {
        mv->requireDataPerObject<real2>(channel_names::areaVolumes, DataManager::PersistenceMode::None);
    }
    else
    {
        die("Internal membrane forces can only be computed with a MembraneVector");
    }
}

void BaseMembraneInteraction::halo(ParticleVector *pv1,
                                   __UNUSED ParticleVector *pv2,
                                   __UNUSED CellList *cl1,
                                   __UNUSED CellList *cl2,
                                   __UNUSED cudaStream_t stream)
{
    debug("Not computing internal membrane forces between local and halo membranes of '%s'",
          pv1->getCName());
}

bool BaseMembraneInteraction::isSelfObjectInteraction() const
{
    return true;
}

void BaseMembraneInteraction::_precomputeQuantities(MembraneVector *mv, cudaStream_t stream)
{
    if (mv->getObjectSize() != mv->mesh->getNvertices())
        die("Object size of '%s' (%d) and number of vertices (%d) mismatch",
            mv->getCName(), mv->getObjectSize(), mv->mesh->getNvertices());

    debug("Computing areas and volumes for %d cells of '%s'",
          mv->local()->getNumObjects(), mv->getCName());

    OVviewWithAreaVolume view(mv, mv->local());

    MembraneMeshView mesh(static_cast<MembraneMesh*>(mv->mesh.get()));

    mv->local()
        ->dataPerObject.getData<real2>(channel_names::areaVolumes)
        ->clearDevice(stream);

    constexpr int nthreads = 128;

    SAFE_KERNEL_LAUNCH(
        base_membrane_interaction_kernels::computeAreaAndVolume,
        view.nObjects, nthreads, 0, stream,
        view, mesh);
}

} // namespace mirheo
