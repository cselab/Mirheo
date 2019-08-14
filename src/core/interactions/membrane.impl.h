#pragma once

#include "interface.h"
#include "membrane/forces_kernels.h"
#include "membrane/parameters.h"
#include "membrane/prerequisites.h"
#include "utils/step_random_gen.h"

#include <core/pvs/membrane_vector.h>
#include <core/pvs/views/ov.h>
#include <core/utils/cuda_common.h>
#include <core/utils/kernel_launch.h>
#include <core/utils/restart_helpers.h>

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
static MembraneForcesKernels::GPU_CommonMembraneParameters setParams(const CommonMembraneParameters& p, StepRandomGen& stepGen, const MirState *state)
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

    InteractionMembraneImpl(const MirState *state, std::string name, CommonMembraneParameters parameters,
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
    
    void local (ParticleVector *pv1,
                __UNUSED ParticleVector *pv2,
                __UNUSED CellList *cl1,
                __UNUSED CellList *cl2,
                cudaStream_t stream) override
    {
        this->precomputeQuantities(pv1, stream);
        
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

    void halo(__UNUSED ParticleVector *pv1,
              __UNUSED ParticleVector *pv2,
              __UNUSED CellList *cl1,
              __UNUSED CellList *cl2,
              __UNUSED cudaStream_t stream) override
    {}

    void setPrerequisites(ParticleVector *pv1, ParticleVector *pv2, CellList *cl1, CellList *cl2) override
    {
        setPrerequisitesPerEnergy(dihedralParams, pv1, pv2, cl1, cl2);
        setPrerequisitesPerEnergy(triangleParams, pv1, pv2, cl1, cl2);
    }

    void precomputeQuantities(ParticleVector *pv1, cudaStream_t stream)
    {
        precomputeQuantitiesPerEnergy(dihedralParams, pv1, stream);
        precomputeQuantitiesPerEnergy(triangleParams, pv1, stream);
    }
    
    void checkpoint(MPI_Comm comm, std::string path, int checkpointId) override
    {
        auto fname = createCheckpointNameWithId(path, "MembraneInt", "txt", checkpointId);
        TextIO::write(fname, stepGen);
        createCheckpointSymlink(comm, path, "MembraneInt", "txt", checkpointId);
    }
    
    void restart(__UNUSED MPI_Comm comm, std::string path) override
    {
        auto fname = createCheckpointName(path, "MembraneInt", "txt");
        auto good = TextIO::read(fname, stepGen);
        if (!good) die("failed to read '%s'\n", fname.c_str());
    }

    
protected:

    std::function< float(float) > scaleFromTime;
    CommonMembraneParameters parameters;
    typename DihedralInteraction::ParametersType dihedralParams;
    typename TriangleInteraction::ParametersType triangleParams;
    StepRandomGen stepGen;
};
