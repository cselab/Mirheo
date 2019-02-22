#include "membrane_WLC_Juelicher.h"
#include "membrane.impl.h"
#include "membrane/common.h"
#include "membrane/dihedral/juelicher.h"

#include <core/utils/make_unique.h>

namespace InteractionMembraneJuelicherKernels
{
__device__ inline float compute_lenTheta(float3 v0, float3 v1, float3 v2, float3 v3)
{
    float len = length(v2 - v0);
    float theta = supplementaryDihedralAngle(v0, v1, v2, v3);
    return len * theta;
}

__global__ void computeAreasAndCurvatures(OVviewWithJuelicherQuants view, MembraneMeshView mesh)
{
    int rbcId = blockIdx.y;
    int idv0  = blockIdx.x * blockDim.x + threadIdx.x;
    int offset = rbcId * mesh.nvertices;

    float lenTheta = 0;
    
    if (idv0 < mesh.nvertices)
    {        
        int startId = mesh.maxDegree * idv0;
        int degree = mesh.degrees[idv0];
        
        int idv1 = mesh.adjacent[startId];
        int idv2 = mesh.adjacent[startId+1];
        
        float3 v0 = fetchPosition(view, offset + idv0);
        float3 v1 = fetchPosition(view, offset + idv1);
        float3 v2 = fetchPosition(view, offset + idv2);
        
        float area = 0;    
        
#pragma unroll 2
        for (int i = 0; i < degree; i++) {
            
            int idv3 = mesh.adjacent[startId + (i+2) % degree];
            float3 v3 = fetchPosition(view, offset + idv3);
            
            area     += 0.3333333f * triangleArea(v0, v1, v2);
            lenTheta += compute_lenTheta(v0, v1, v2, v3);
            
            v1 = v2;
            v2 = v3;
        }
        
        view.vertexAreas          [offset + idv0] = area;
        view.vertexMeanCurvatures [offset + idv0] = lenTheta / (4 * area);
    }
    
    lenTheta = warpReduce( lenTheta, [] (float a, float b) { return a+b; } );

    if (__laneid() == 0)
        atomicAdd(&view.lenThetaTot[rbcId], lenTheta);
}
} // namespace InteractionMembraneJuelicherKernels

InteractionMembraneWLCJuelicher::InteractionMembraneWLCJuelicher(const YmrState *state, std::string name,
                                                                 MembraneParameters parameters, JuelicherBendingParameters juelicherParams,
                                                                 bool stressFree, float growUntil) :
    InteractionMembrane(state, name)
{
    impl = std::make_unique<InteractionMembraneImpl<DihedralJuelicher>>(state, name, parameters, juelicherParams, stressFree, growUntil);
}

InteractionMembraneWLCJuelicher::~InteractionMembraneWLCJuelicher() = default;

void InteractionMembraneWLCJuelicher::setPrerequisites(ParticleVector *pv1, ParticleVector *pv2, CellList *cl1, CellList *cl2)
{
    InteractionMembrane::setPrerequisites(pv1, pv2, cl1, cl2);

    auto ov = dynamic_cast<MembraneVector*>(pv1);
    
    ov->requireDataPerObject<float>(ChannelNames::lenThetaTot, ExtraDataManager::CommunicationMode::None, ExtraDataManager::PersistenceMode::None);

    ov->requireDataPerParticle<float>(ChannelNames::areas, ExtraDataManager::CommunicationMode::None, ExtraDataManager::PersistenceMode::None);
    ov->requireDataPerParticle<float>(ChannelNames::meanCurvatures, ExtraDataManager::CommunicationMode::None, ExtraDataManager::PersistenceMode::None);
}

void InteractionMembraneWLCJuelicher::precomputeQuantities(ParticleVector *pv1, cudaStream_t stream)
{
    auto ov = dynamic_cast<MembraneVector *>(pv1);

    debug("Computing vertex areas and curvatures for %d cells of '%s'",
          ov->local()->nObjects, ov->name.c_str());

    OVviewWithJuelicherQuants view(ov, ov->local());

    MembraneMeshView mesh(static_cast<MembraneMesh*>(ov->mesh.get()));

    const int nthreads = 128;    

    dim3 threads(nthreads, 1);
    dim3 blocks(getNblocks(mesh.nvertices, nthreads), view.nObjects);
        
    SAFE_KERNEL_LAUNCH(
        InteractionMembraneJuelicherKernels::computeAreasAndCurvatures,
        blocks, threads, 0, stream,
        view, mesh );
}
