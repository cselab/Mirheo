#pragma once

#include "interface.h"
#include "membrane/forces_kernels.h"
#include "membrane/parameters.h"

#include <core/pvs/membrane_vector.h>
#include <core/pvs/views/ov.h>
#include <core/utils/cuda_common.h>
#include <core/utils/kernel_launch.h>

#include <cmath>
#include <functional>
#include <random>

/**
 * Provide mapping from the model parameters to the parameters
 * used on GPU to compute the forces.
 *
 * @param p model parameters
 * @param m RBC membrane mesh
 * @return parameters to be passed to GPU kernels
 */
static MembraneForcesKernels::GPU_RBCparameters setParams(const MembraneParameters& p, float dt, float t)
{
    MembraneForcesKernels::GPU_RBCparameters devP;

    devP.gammaC = p.gammaC;
    devP.gammaT = p.gammaT;

    devP.totArea0   = p.totArea0;
    devP.totVolume0 = p.totVolume0;

    devP.ka0 = p.ka / p.totArea0;
    devP.kv0 = p.kv / (6.0*p.totVolume0);

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

/**
 * Generic mplementation of RBC membrane forces
 */
template <class TriangleInteraction, class DihedralInteraction>
class InteractionMembraneImpl : public Interaction
{
public:

    InteractionMembraneImpl(const YmrState *state, std::string name, MembraneParameters parameters,
                            typename TriangleInteraction::ParametersType triangleParams,
                            typename DihedralInteraction::ParametersType dihedralParams,
                            float growUntil) :
        Interaction(state, name, 1.0f),
        parameters(parameters),
        scaleFromTime( [growUntil] (float t) { return min(1.0f, 0.5f + 0.5f * (t / growUntil)); } ),
        dihedralParams(dihedralParams),
        triangleParams(triangleParams)
    {}

    ~InteractionMembraneImpl() = default;
    
    void local (ParticleVector *pv1, ParticleVector *pv2, CellList *cl1, CellList *cl2, cudaStream_t stream) override
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
        auto mesh = static_cast<MembraneMesh *>(ov->mesh.get());
        MembraneMeshView meshView(mesh);

        const int nthreads = 128;
        const int nblocks  = getNblocks(view.size, nthreads);

        auto devParams = setParams(currentParams, state->dt, state->currentTime);

        DihedralInteraction dihedralInteraction(dihedralParams, scale);
        TriangleInteraction triangleInteraction(triangleParams, mesh, scale);

        devParams.scale = scale;

        SAFE_KERNEL_LAUNCH(MembraneForcesKernels::computeMembraneForces,
                           nblocks, nthreads, 0, stream,
                           triangleInteraction,
                           dihedralInteraction, dihedralView,
                           view, meshView, devParams);

    }

    void halo(ParticleVector *pv1, ParticleVector *pv2, CellList *cl1, CellList *cl2, cudaStream_t stream) {}
    
protected:

    std::function< float(float) > scaleFromTime;
    MembraneParameters parameters;
    typename DihedralInteraction::ParametersType dihedralParams;
    typename TriangleInteraction::ParametersType triangleParams;
};
