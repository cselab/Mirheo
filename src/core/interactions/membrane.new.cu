#include "membrane.new.h"

#include "membrane/common.h"

#include <core/pvs/membrane_vector.h>
#include <core/pvs/views/ov.h>
#include <core/utils/cuda_common.h>
#include <core/utils/kernel_launch.h>


namespace MembraneKernels
{
__global__ void computeAreaAndVolume(OVviewWithAreaVolume view, MeshView mesh)
{
    int objId = blockIdx.x;
    int offset = objId * mesh.nvertices;
    float2 a_v = make_float2(0.0f);

    for(int i = threadIdx.x; i < mesh.ntriangles; i += blockDim.x) {        
        int3 ids = mesh.triangles[i];

        float3 v0 = f4tof3( view.particles[ 2 * (offset + ids.x) ] );
        float3 v1 = f4tof3( view.particles[ 2 * (offset + ids.y) ] );
        float3 v2 = f4tof3( view.particles[ 2 * (offset + ids.z) ] );

        a_v.x += triangleArea(v0, v1, v2);
        a_v.y += triangleSignedVolume(v0, v1, v2);
    }

    a_v = warpReduce( a_v, [] (float a, float b) { return a+b; } );

    if (__laneid() == 0)
        atomicAdd(&view.area_volumes[objId], a_v);
}
} // namespace MembraneKernels

InteractionMembraneNew::InteractionMembraneNew(const YmrState *state, std::string name) :
    Interaction(state, name, /* default cutoff rc */ 1.0),
    impl(nullptr)
{}

InteractionMembraneNew::~InteractionMembraneNew() = default;

void InteractionMembraneNew::setPrerequisites(ParticleVector *pv1, ParticleVector *pv2, CellList *cl1, CellList *cl2)
{
    if (pv1 != pv2)
        die("Internal membrane forces can't be computed between two different particle vectors");

    auto ov = dynamic_cast<MembraneVector*>(pv1);
    if (ov == nullptr)
        die("Internal membrane forces can only be computed with a MembraneVector");

    ov->requireDataPerObject<float2>(ChannelNames::areaVolumes,
                                     ExtraDataManager::CommunicationMode::None,
                                     ExtraDataManager::PersistenceMode::None);
}

void InteractionMembraneNew::local(ParticleVector *pv1, ParticleVector *pv2, CellList *cl1, CellList *cl2, cudaStream_t stream)
{
    precomputeQuantities(pv1, stream);
    impl->local(pv1, pv2, cl1, cl2, stream);
}

void InteractionMembraneNew::halo(ParticleVector *pv1, ParticleVector *pv2, CellList *cl1, CellList *cl2, cudaStream_t stream)
{
    debug("Not computing internal membrane forces between local and halo membranes of '%s'",
          pv1->name.c_str());
}
    
void InteractionMembraneNew::precomputeQuantities(ParticleVector *pv1, cudaStream_t stream)
{
    auto ov = dynamic_cast<MembraneVector *>(pv1);

    if (ov->objSize != ov->mesh->getNvertices())
        die("Object size of '%s' (%d) and number of vertices (%d) mismatch",
            ov->name.c_str(), ov->objSize, ov->mesh->getNvertices());

    debug("Computing areas and volumes for %d cells of '%s'",
          ov->local()->nObjects, ov->name.c_str());

    OVviewWithAreaVolume view(ov, ov->local());

    MembraneMeshView mesh(static_cast<MembraneMesh*>(ov->mesh.get()));

    ov->local()
        ->extraPerObject.getData<float2>(ChannelNames::areaVolumes)
        ->clearDevice(stream);
    
    const int nthreads = 128;
    SAFE_KERNEL_LAUNCH(MembraneKernels::computeAreaAndVolume,
                       view.nObjects, nthreads, 0, stream,
                       view, mesh);
}
