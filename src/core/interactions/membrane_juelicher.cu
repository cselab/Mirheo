#include <core/pvs/membrane_vector.h>
#include <core/pvs/views/ov.h>
#include <core/utils/cuda_common.h>
#include <core/utils/kernel_launch.h>

#include "membrane_juelicher.h"
#include "membrane/bending_juelicher.h"

InteractionMembraneJuelicher::InteractionMembraneJuelicher(const YmrState *state, std::string name,
                                                           MembraneParameters parameters,
                                                           JuelicherBendingParameters bendingParameters,
                                                           bool stressFree, float growUntil) :
    InteractionMembrane(state, name, parameters, stressFree, growUntil),
    bendingParameters(bendingParameters)
{}


InteractionMembraneJuelicher::~InteractionMembraneJuelicher() = default;
    
void InteractionMembraneJuelicher::setPrerequisites(ParticleVector *pv1, ParticleVector *pv2, CellList *cl1, CellList *cl2)
{
    InteractionMembrane::setPrerequisites(pv1, pv2, cl1, cl2);

    auto ov = dynamic_cast<MembraneVector*>(pv1);
    
    ov->requireDataPerObject<float>(ChannelNames::lenThetaTot, ExtraDataManager::CommunicationMode::None, ExtraDataManager::PersistenceMode::None);

    ov->requireDataPerParticle<float>(ChannelNames::areas, ExtraDataManager::CommunicationMode::None, ExtraDataManager::PersistenceMode::None);
    ov->requireDataPerParticle<float>(ChannelNames::meanCurvatures, ExtraDataManager::CommunicationMode::None, ExtraDataManager::PersistenceMode::None);
}


static BendingJuelicherKernels::GPU_BendingParams setJuelicherBendingParams(float scale, JuelicherBendingParameters& p)
{
    BendingJuelicherKernels::GPU_BendingParams devP;

    devP.kb     = p.kb  * scale*scale;
    devP.kad_pi = p.kad * M_PI * scale*scale;

    devP.H0  = p.C0 / (2 * scale);
    devP.DA0 = p.DA0 * scale*scale;

    return devP;
}

void InteractionMembraneJuelicher::bendingForces(float scale, MembraneVector *ov, MembraneMeshView mesh, cudaStream_t stream)
{
    ov->local()->extraPerObject.getData<float>(ChannelNames::lenThetaTot)->clearDevice(stream);

    OVviewWithJuelicherQuants view(ov, ov->local());
    
    const int nthreads = 128;    

    {
        dim3 threads(nthreads, 1);
        dim3 blocks(getNblocks(mesh.nvertices, nthreads), view.nObjects);
        
        SAFE_KERNEL_LAUNCH(
            BendingJuelicherKernels::computeAreasAndCurvatures,
            blocks, threads, 0, stream,
            view, mesh );
    }

    {
        auto devParams = setJuelicherBendingParams(scale, bendingParameters);
        
        const int blocks = getNblocks(view.size, nthreads);
    
        SAFE_KERNEL_LAUNCH(
            BendingJuelicherKernels::computeBendingForces,
            blocks, nthreads, 0, stream,
            view, mesh, devParams );
    }
}
