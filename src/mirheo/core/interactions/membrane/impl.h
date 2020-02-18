#pragma once

#include "drivers.h"
#include "force_kernels/parameters.h"
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
        Interaction(state, name),
        parameters_(parameters),
        growUntil_(growUntil),
        dihedralParams_(dihedralParams),
        triangleParams_(triangleParams),
        filter_(filter),
        stepGen_(seed)
    {}

    /** \brief Construct the interaction object from a snapshot.
        \param [in] state The global state of the system.
        \param [in] loader The \c Loader object. Provides load context and unserialization functions.
        \param [in] config The parameters of the interaction.
     */
    MembraneInteractionImpl(const MirState *state, Loader& loader, const ConfigObject& config) :
        Interaction(state, loader, config),
        growUntil_{config["growUntil"]},
        parameters_{loader.load<CommonMembraneParameters>(config["parameters"])},
        dihedralParams_{loader.load<typename DihedralInteraction::ParametersType>(config["dihedralParams"])},
        triangleParams_{loader.load<typename TriangleInteraction::ParametersType>(config["triangleParams"])},
        filter_{loader.load<Filter>(config["filter"])},
        stepGen_(42424242)
    {
        warn("stepGen save/load not imported, resetting the seed!");
    }

    ~MembraneInteractionImpl() = default;

    real scaleFromTime(real t) const {
        return math::min(1.0_r, 0.5_r + 0.5_r * (t / growUntil_));
    }
    
    void local(ParticleVector *pv1,
               __UNUSED ParticleVector *pv2,
               __UNUSED CellList *cl1,
               __UNUSED CellList *cl2,
               cudaStream_t stream) override
    {
        this->precomputeQuantities(pv1, stream);
        
        auto ov = dynamic_cast<MembraneVector *>(pv1);

        if (ov->getObjectSize() != ov->mesh->getNvertices())
            die("Object size of '%s' (%d) and number of vertices (%d) mismatch",
                ov->getCName(), ov->getObjectSize(), ov->mesh->getNvertices());

        debug("Computing internal membrane forces for %d cells of '%s'",
              ov->local()->getNumObjects(), ov->getCName());

        auto currentParams = parameters_;
        const real scale = scaleFromTime(getState()->currentTime);
        rescaleParameters(currentParams, scale);

        OVviewWithAreaVolume view(ov, ov->local());
        typename DihedralInteraction::ViewType dihedralView(ov, ov->local());
        auto mesh = static_cast<MembraneMesh *>(ov->mesh.get());
        MembraneMeshView meshView(mesh);

        const int nthreads = 128;
        const int nblocks  = getNblocks(view.size, nthreads);

        const auto devParams = setParams(currentParams, stepGen_, getState());

        DihedralInteraction dihedralInteraction(dihedralParams_, scale);
        TriangleInteraction triangleInteraction(triangleParams_, mesh, scale);
        filter_.setup(ov);
        
        SAFE_KERNEL_LAUNCH(MembraneForcesKernels::computeMembraneForces,
                           nblocks, nthreads, 0, stream,
                           triangleInteraction,
                           dihedralInteraction, dihedralView,
                           view, meshView, devParams, filter_);

    }

    void halo(__UNUSED ParticleVector *pv1,
              __UNUSED ParticleVector *pv2,
              __UNUSED CellList *cl1,
              __UNUSED CellList *cl2,
              __UNUSED cudaStream_t stream) override
    {}

    void setPrerequisites(ParticleVector *pv1,
                          __UNUSED ParticleVector *pv2,
                          __UNUSED CellList *cl1,
                          __UNUSED CellList *cl2) override
    {
        if (auto mv = dynamic_cast<MembraneVector*>(pv1))
        {
            setPrerequisitesPerEnergy(dihedralParams_, mv);
            setPrerequisitesPerEnergy(triangleParams_, mv);
            filter_.setPrerequisites(mv);
        }
        else
        {
            die("Interaction '%s' needs a membrane vector (given '%s')",
                this->getCName(), pv1->getCName());
        }
    }

    void precomputeQuantities(ParticleVector *pv, cudaStream_t stream)
    {
        if (auto mv = dynamic_cast<MembraneVector*>(pv))
        {
            precomputeQuantitiesPerEnergy(dihedralParams_, mv, stream);
            precomputeQuantitiesPerEnergy(triangleParams_, mv, stream);
        }
        else
        {
            die("%s is not a Membrane Vector", pv->getCName());
        }
    }
    
    void checkpoint(MPI_Comm comm, const std::string& path, int checkpointId) override
    {
        const auto fname = createCheckpointNameWithId(path, "MembraneInt", "txt", checkpointId);
        TextIO::write(fname, stepGen_);
        createCheckpointSymlink(comm, path, "MembraneInt", "txt", checkpointId);
    }
    
    void restart(__UNUSED MPI_Comm comm, const std::string& path) override
    {
        const auto fname = createCheckpointName(path, "MembraneInt", "txt");
        const bool good = TextIO::read(fname, stepGen_);
        if (!good) die("failed to read '%s'\n", fname.c_str());
    }

    static std::string getTypeName()
    {
        return constructTypeName<TriangleInteraction, DihedralInteraction, Filter>(
                "MembraneInteractionImpl");
    }
    void saveSnapshotAndRegister(Saver& saver)
    {
        saver.registerObject<MembraneInteractionImpl>(this, _saveSnapshot(saver, getTypeName()));
    }

protected:
    /** \brief Implementation of snapshot saving. Reusable by potential derived classes.
        \param [in,out] saver The \c Saver object. Provides save context and serialization functions.
        \param [in] typeName The name of the type being saved.
      */
    ConfigObject _saveSnapshot(Saver& saver, const std::string& typeName)
    {
        ConfigObject config = Interaction::_saveImplSnapshot(saver, typeName);
        config.emplace("growUntil",      saver(growUntil_));
        config.emplace("parameters",     saver(parameters_));
        config.emplace("dihedralParams", saver(dihedralParams_));
        config.emplace("triangleParams", saver(triangleParams_));
        config.emplace("filter",         saver(filter_));
        config.emplace("stepGen",        saver("<<not implemented>>"));
        return config;
    }

    real growUntil_;
    CommonMembraneParameters parameters_;
    typename DihedralInteraction::ParametersType dihedralParams_;
    typename TriangleInteraction::ParametersType triangleParams_;
    Filter filter_;
    StepRandomGen stepGen_;
};

} // namespace mirheo
