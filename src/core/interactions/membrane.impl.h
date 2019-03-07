#pragma once

#include "interface.h"
#include "membrane/forces_kernels.h"
#include "membrane/parameters.h"
#include "utils/step_random_gen.h"

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
static MembraneForcesKernels::GPU_CommonMembraneParameters setParams(const CommonMembraneParameters& p, StepRandomGen& stepGen, const YmrState *state)
{
    MembraneForcesKernels::GPU_CommonMembraneParameters devP;

    devP.gammaC = p.gammaC;
    devP.gammaT = p.gammaT;

    devP.totArea0   = p.totArea0;
    devP.totVolume0 = p.totVolume0;

    devP.ka0 = p.ka / p.totArea0;
    devP.kv0 = p.kv / (6.0*p.totVolume0);

    devP.fluctuationForces = p.fluctuationForces;

    if (devP.fluctuationForces) {
        float dt = state->dt;
        devP.seed = stepGen.generate(state);
        devP.sigma_rnd = sqrt(2 * p.kBT * p.gammaC / dt);
    }
    
    return devP;
}

static void rescaleParameters(CommonMembraneParameters& p, float scale)
{
    p.totArea0   *= scale * scale;
    p.totVolume0 *= scale * scale * scale;
    p.kBT        *= scale * scale;

    p.gammaC *= scale;
    p.gammaT *= scale;
}

/**
 * Generic mplementation of RBC membrane forces
 */
template <class TriangleInteraction, class DihedralInteraction>
class InteractionMembraneImpl : public Interaction
{
public:

    InteractionMembraneImpl(const YmrState *state, std::string name, CommonMembraneParameters parameters,
                            typename TriangleInteraction::ParametersType triangleParams,
                            typename DihedralInteraction::ParametersType dihedralParams,
                            float growUntil, long seed = 42424242) :
        Interaction(state, name, 1.0f),
        parameters(parameters),
        scaleFromTime( [growUntil] (float t) { return min(1.0f, 0.5f + 0.5f * (t / growUntil)); } ),
        dihedralParams(dihedralParams),
        triangleParams(triangleParams),
        stepGen(seed)
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
        rescaleParameters(currentParams, scale);

        OVviewWithAreaVolume view(ov, ov->local());
        typename DihedralInteraction::ViewType dihedralView(ov, ov->local());
        auto mesh = static_cast<MembraneMesh *>(ov->mesh.get());
        MembraneMeshView meshView(mesh);

        const int nthreads = 128;
        const int nblocks  = getNblocks(view.size, nthreads);

        auto devParams = setParams(currentParams, stepGen, state);

        DihedralInteraction dihedralInteraction(dihedralParams, scale);
        TriangleInteraction triangleInteraction(triangleParams, mesh, scale);

        SAFE_KERNEL_LAUNCH(MembraneForcesKernels::computeMembraneForces,
                           nblocks, nthreads, 0, stream,
                           triangleInteraction,
                           dihedralInteraction, dihedralView,
                           view, meshView, devParams);

    }

    void halo(ParticleVector *pv1, ParticleVector *pv2, CellList *cl1, CellList *cl2, cudaStream_t stream) {}
    
protected:

    std::function< float(float) > scaleFromTime;
    CommonMembraneParameters parameters;
    typename DihedralInteraction::ParametersType dihedralParams;
    typename TriangleInteraction::ParametersType triangleParams;
    StepRandomGen stepGen;
};
