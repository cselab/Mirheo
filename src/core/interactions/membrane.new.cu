#include "membrane.new.h"
#include "membrane/forces_kernels.h"

#include "membrane/dihedral/kantor.h"
#include "membrane/dihedral/juelicher.h"

#include <core/pvs/membrane_vector.h>
#include <core/pvs/views/ov.h>
#include <core/utils/cuda_common.h>
#include <core/utils/kernel_launch.h>

#include <cmath>
#include <random>

/**
 * Provide mapping from the model parameters to the parameters
 * used on GPU to compute the forces.
 *
 * @param p model parameters
 * @param m RBC membrane mesh
 * @return parameters to be passed to GPU kernels
 */
static MembraneForcesKernels::GPU_RBCparameters setParams(MembraneParameters& p, Mesh *m, float dt, float t)
{
    MembraneForcesKernels::GPU_RBCparameters devP;

    devP.gammaC = p.gammaC;
    devP.gammaT = p.gammaT;

    devP.area0 = p.totArea0 / m->getNtriangles();
    devP.totArea0 = p.totArea0;
    devP.totVolume0 = p.totVolume0;

    devP.x0   = p.x0;
    devP.ks   = p.ks;
    devP.mpow = p.mpow;
    devP.l0 = sqrt(devP.area0 * 4.0 / sqrt(3.0));    

    devP.ka0 = p.ka / p.totArea0;
    devP.kv0 = p.kv / (6.0*p.totVolume0);
    devP.kd0 = p.kd;

    devP.fluctuationForces = p.fluctuationForces;

    if (devP.fluctuationForces) {
        int v = *((int*)&t);
        std::mt19937 gen(v);
        std::uniform_real_distribution<float> udistr(0.001, 1);
        devP.seed = udistr(gen);
        devP.sigma_rnd = sqrt(2 * p.kbT * p.gammaC / dt);
    }
    
    return devP;
}

template <class DihedralInteraction>
InteractionMembrane<DihedralInteraction>::
InteractionMembrane(const YmrState *state, std::string name, MembraneParameters parameters,
                    DihedralInteraction dihedralInteraction, bool stressFree, float growUntil) :
    Interaction(state, name, 1.0f),
    parameters(parameters),
    stressFree(stressFree),
    scaleFromTime( [growUntil] (float t) { return min(1.0f, 0.5f + 0.5f * (t / growUntil)); } ),
    dihedralInteraction(dihedralInteraction)
{}

template <class DihedralInteraction>
InteractionMembrane<DihedralInteraction>::
~InteractionMembrane() = default;

/**
 * Require that \p pv1 and \p pv2 are the same and are instances
 * of MembraneVector
 */
template <class DihedralInteraction>
void InteractionMembrane<DihedralInteraction>::
setPrerequisites(ParticleVector *pv1, ParticleVector *pv2, CellList *cl1, CellList *cl2)
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

/**
 * Compute the membrane forces.
 *
 * First call the computeAreaAndVolume() kernel to compute area
 * and volume of each cell, then use these data to calculate the
 * forces themselves by calling computeMembraneForces() kernel
 */
template <class DihedralInteraction>
void InteractionMembrane<DihedralInteraction>::
local(ParticleVector *pv1, ParticleVector *pv2,
      CellList *cl1, CellList *cl2,
      cudaStream_t stream)
{
    auto ov = dynamic_cast<MembraneVector *>(pv1);

    if (ov->objSize != ov->mesh->getNvertices())
        die("Object size of '%s' (%d) and number of vertices (%d) mismatch",
            ov->name.c_str(), ov->objSize, ov->mesh->getNvertices());

    debug("Computing internal membrane forces for %d cells of '%s'",
          ov->local()->nObjects, ov->name.c_str());

    auto currentParams = parameters;
    float scale = scaleFromTime(state->currentTime);
    currentParams.totArea0 *= scale * scale;
    currentParams.totVolume0 *= scale * scale * scale;
    currentParams.kbT *= scale * scale;
    currentParams.ks *= scale * scale;

    currentParams.gammaC *= scale;
    currentParams.gammaT *= scale;

    OVviewWithAreaVolume view(ov, ov->local());
    typename DihedralInteraction::ViewType dihedralView(ov, ov->local());
    MembraneMeshView mesh(static_cast<MembraneMesh *>(ov->mesh.get()));
    ov->local()
        ->extraPerObject.getData<float2>(ChannelNames::areaVolumes)
        ->clearDevice(stream);

    const int nthreads = 128;
    SAFE_KERNEL_LAUNCH(
        MembraneForcesKernels::computeAreaAndVolume,
        view.nObjects, nthreads, 0, stream,
        view, mesh);

    const int blocks = getNblocks(view.size, nthreads);

    auto devParams = setParams(currentParams, ov->mesh.get(), state->dt, state->currentTime);

    devParams.scale = scale;

    SAFE_KERNEL_LAUNCH(
        MembraneForcesKernels::computeMembraneForces,
        blocks, nthreads, 0, stream,
        stressFree, this->dihedralInteraction, dihedralView,
        view, mesh, devParams);
}

template <class DihedralInteraction>
void InteractionMembrane<DihedralInteraction>::
halo(ParticleVector *pv1, ParticleVector *pv2,
     CellList *cl1, CellList *cl2, cudaStream_t stream)
{
    debug("Not computing internal membrane forces between local and halo membranes of '%s'",
          pv1->name.c_str());
}


template class InteractionMembrane<DihedralKantor>;
template class InteractionMembrane<DihedralJuelicher>;