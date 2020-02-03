#pragma once

#include "drivers.h"
#include "kernels/parameters.h"
#include "prerequisites.h"

#include <mirheo/core/interactions/interface.h>
#include <mirheo/core/interactions/utils/step_random_gen.h>
#include <mirheo/core/pvs/membrane_vector.h>
#include <mirheo/core/pvs/views/ov.h>
#include <mirheo/core/utils/config.h>
#include <mirheo/core/utils/cuda_common.h>
#include <mirheo/core/utils/kernel_launch.h>
#include <mirheo/core/utils/restart_helpers.h>

#include <cmath>

namespace mirheo
{

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
    devP.kv0 = p.kv / (6.0_r*p.totVolume0);

    devP.fluctuationForces = p.fluctuationForces;

    if (devP.fluctuationForces)
    {
        const auto dt = state->dt;
        devP.seed = stepGen.generate(state);
        devP.sigma_rnd = math::sqrt(2 * p.kBT * p.gammaC / dt);
    }
    
    return devP;
}

static void rescaleParameters(CommonMembraneParameters& p, real scale)
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
template <class TriangleInteraction, class DihedralInteraction, class Filter>
class MembraneInteractionImpl : public Interaction
{
public:

    MembraneInteractionImpl(const MirState *state, std::string name, CommonMembraneParameters parameters,
                            typename TriangleInteraction::ParametersType triangleParams,
                            typename DihedralInteraction::ParametersType dihedralParams,
                            real growUntil, Filter filter, long seed = 42424242) :
        Interaction(state, name, 1.0_r),
        parameters(parameters),
        growUntil(growUntil),
        dihedralParams(dihedralParams),
        triangleParams(triangleParams),
        filter(filter),
        stepGen(seed)
    {}

    ~MembraneInteractionImpl() = default;
    Config getConfig() const override {
        return Config::Dictionary{
            {"name", name},
            {"rc", rc},
            {"growUntil", growUntil},
            {"parameters", parameters},
            {"dihedralParams", dihedralParams},
            {"triangleParams", triangleParams},
            {"filter", filter},
            {"stepGen", std::string("<<not implemented>")},
        };
    }

    real scaleFromTime(real t) const {
        return math::min(1.0_r, 0.5_r + 0.5_r * (t / growUntil));
    }
    
    void local(ParticleVector *pv1,
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
        const real scale = scaleFromTime(getState()->currentTime);
        rescaleParameters(currentParams, scale);

        OVviewWithAreaVolume view(ov, ov->local());
        typename DihedralInteraction::ViewType dihedralView(ov, ov->local());
        auto mesh = static_cast<MembraneMesh *>(ov->mesh.get());
        MembraneMeshView meshView(mesh);

        const int nthreads = 128;
        const int nblocks  = getNblocks(view.size, nthreads);

        const auto devParams = setParams(currentParams, stepGen, getState());

        DihedralInteraction dihedralInteraction(dihedralParams, scale);
        TriangleInteraction triangleInteraction(triangleParams, mesh, scale);
        filter.setup(ov);
        
        SAFE_KERNEL_LAUNCH(MembraneForcesKernels::computeMembraneForces,
                           nblocks, nthreads, 0, stream,
                           triangleInteraction,
                           dihedralInteraction, dihedralView,
                           view, meshView, devParams, filter);

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

        if (auto mv = dynamic_cast<MembraneVector*>(pv1))
            filter.setPrerequisites(mv);
        else
            die("Interaction '%s' needs a membrane vector (given '%s')",
                this->name.c_str(), pv1->name.c_str());
    }

    void precomputeQuantities(ParticleVector *pv1, cudaStream_t stream)
    {
        precomputeQuantitiesPerEnergy(dihedralParams, pv1, stream);
        precomputeQuantitiesPerEnergy(triangleParams, pv1, stream);
    }
    
    void checkpoint(MPI_Comm comm, const std::string& path, int checkpointId) override
    {
        const auto fname = createCheckpointNameWithId(path, "MembraneInt", "txt", checkpointId);
        TextIO::write(fname, stepGen);
        createCheckpointSymlink(comm, path, "MembraneInt", "txt", checkpointId);
    }
    
    void restart(__UNUSED MPI_Comm comm, const std::string& path) override
    {
        const auto fname = createCheckpointName(path, "MembraneInt", "txt");
        const bool good = TextIO::read(fname, stepGen);
        if (!good) die("failed to read '%s'\n", fname.c_str());
    }

    
protected:

    real growUntil;
    CommonMembraneParameters parameters;
    typename DihedralInteraction::ParametersType dihedralParams;
    typename TriangleInteraction::ParametersType triangleParams;
    Filter filter;
    StepRandomGen stepGen;
    
};

} // namespace mirheo
