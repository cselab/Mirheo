// Copyright 2020 ETH Zurich. All Rights Reserved.
#pragma once

#include "base_membrane.h"
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

namespace mirheo
{

/** \brief Extract the constraint parameters to a struct usable on device.
    \param [in] p model parameters
    \return parameters to be passed to GPU kernels
 */
static membrane_forces_kernels::GPUConstraintMembraneParameters getConstraintParams(const CommonMembraneParameters& p)
{
    membrane_forces_kernels::GPUConstraintMembraneParameters devP;

    devP.totArea0   = p.totArea0;
    devP.totVolume0 = p.totVolume0;

    devP.ka0 = p.ka / p.totArea0;
    devP.kv0 = p.kv / (6.0_r*p.totVolume0);

    return devP;
}

/** \brief Convert model parameters struct to a struct usable on device.
    \param [in] p model parameters
    \param [in,out] stepGen Random number generator
    \param [in] state Simulation state
    \return parameters to be passed to GPU kernels
 */
static membrane_forces_kernels::GPUViscMembraneParameters getViscParams(const CommonMembraneParameters& p, StepRandomGen& stepGen, const MirState *state)
{
    membrane_forces_kernels::GPUViscMembraneParameters devP;

    devP.gammaC = p.gammaC;
    devP.gammaT = p.gammaT;

    devP.seed = stepGen.generate(state);

    const auto dt = state->getDt();
    devP.sigma_rnd = math::sqrt(2 * p.kBT * p.gammaC / dt);

    return devP;
}

/** \brief recale parameters for a given length scale
    \param [in,out] p The parameters to rescale in length
    \param [in] scale The scaling factor to transform lengths
 */
static void rescaleParameters(CommonMembraneParameters& p, real scale)
{
    p.totArea0   *= scale * scale;
    p.totVolume0 *= scale * scale * scale;
    p.kBT        *= scale * scale;

    p.gammaC *= scale;
    p.gammaT *= scale;
}

/** \brief Generic implementation of membrane forces.
    \tparam TriangleInteraction Describes what forces are applied to triangles
    \tparam DihedralInteraction Describes what forces are applied to dihedrals
    \tparam Filter This allows to apply the interactions only to a subset of membranes
 */
template <class TriangleInteraction, class DihedralInteraction, class Filter>
class MembraneInteraction : public BaseMembraneInteraction
{
public:

    /** \brief Construct a MembraneInteraction object
        \param [in] state The global state of the system
        \param [in] name The name of the interaction
        \param [in] parameters The common parameters from all kernel forces
        \param [in] triangleParams Parameters that contain the parameters of the triangle forces kernel
        \param [in] dihedralParams Parameters that contain the parameters of the dihedral forces kernel
        \param [in] initLengthFraction The membrane will grow from this fraction of its size to its full size in \p growUntil time
        \param [in] growUntil The membrane will grow from \p initLengthFraction fraction of its size to its full size in this amount of time
        \param [in] filter Describes which membranes to apply the interactions
        \param [in] seed Random seed for rng

        More information can be found on \p growUntil in _scaleFromTime().
    */
    MembraneInteraction(const MirState *state, std::string name, CommonMembraneParameters parameters,
                        typename TriangleInteraction::ParametersType triangleParams,
                        typename DihedralInteraction::ParametersType dihedralParams,
                        real initLengthFraction, real growUntil, Filter filter, long seed = 42424242) :
        BaseMembraneInteraction(state, name),
        parameters_(parameters),
        initLengthFraction_(initLengthFraction),
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
    MembraneInteraction(const MirState *state, Loader& loader, const ConfigObject& config) :
        BaseMembraneInteraction(state, loader, config),
        initLengthFraction_{config["initLengthFraction"]},
        growUntil_{config["growUntil"]},
        parameters_{loader.load<CommonMembraneParameters>(config["parameters"])},
        dihedralParams_{loader.load<typename DihedralInteraction::ParametersType>(config["dihedralParams"])},
        triangleParams_{loader.load<typename TriangleInteraction::ParametersType>(config["triangleParams"])},
        filter_{loader.load<Filter>(config["filter"])},
        stepGen_(42424242)
    {
        warn("stepGen save/load not imported, resetting the seed!");
    }

    ~MembraneInteraction() = default;

    void local(ParticleVector *pv1,
               __UNUSED ParticleVector *pv2,
               __UNUSED CellList *cl1,
               __UNUSED CellList *cl2,
               cudaStream_t stream) override
    {
        auto mv = dynamic_cast<MembraneVector *>(pv1);

        this->_precomputeQuantities(mv, stream);

        if (mv->getObjectSize() != mv->mesh->getNvertices())
            die("Object size of '%s' (%d) and number of vertices (%d) mismatch",
                mv->getCName(), mv->getObjectSize(), mv->mesh->getNvertices());

        debug("Computing internal membrane forces for %d cells of '%s'",
              mv->local()->getNumObjects(), mv->getCName());

        auto currentParams = parameters_;
        const real scale = _scaleFromTime(getState()->currentTime);
        rescaleParameters(currentParams, scale);

        OVviewWithAreaVolume view(mv, mv->local());
        typename DihedralInteraction::ViewType dihedralView(mv, mv->local());
        auto mesh = static_cast<MembraneMesh *>(mv->mesh.get());
        MembraneMeshView meshView(mesh);

        const int nthreads = 128;
        const int nblocks  = getNblocks(view.size, nthreads);

        const auto devConstraintParams = getConstraintParams(currentParams);

        DihedralInteraction dihedralInteraction(dihedralParams_, scale);
        TriangleInteraction triangleInteraction(triangleParams_, mesh, scale);
        filter_.setup(mv);

        SAFE_KERNEL_LAUNCH(
            membrane_forces_kernels::computeMembraneForces,
            nblocks, nthreads, 0, stream,
            triangleInteraction,
            dihedralInteraction, dihedralView,
            view, meshView, devConstraintParams, filter_);

        const auto devViscParams = getViscParams(currentParams, stepGen_, getState());

        if (devViscParams.sigma_rnd == 0 &&
            devViscParams.gammaC == 0 &&
            devViscParams.gammaT == 0)
            return;

        SAFE_KERNEL_LAUNCH(
            membrane_forces_kernels::computeMembraneViscousFluctForces,
            nblocks, nthreads, 0, stream,
            view, meshView, devViscParams, filter_);
    }

    void setPrerequisites(ParticleVector *pv1,
                          __UNUSED ParticleVector *pv2,
                          __UNUSED CellList *cl1,
                          __UNUSED CellList *cl2) override
    {
        BaseMembraneInteraction::setPrerequisites(pv1, pv2, cl1, cl2);

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

    void checkpoint(MPI_Comm comm, const std::string& path, int checkpointId) override
    {
        const auto fname = createCheckpointNameWithId(path, "MembraneInt", "txt", checkpointId);
        text_IO::write(fname, stepGen_);
        createCheckpointSymlink(comm, path, "MembraneInt", "txt", checkpointId);
    }

    void restart(__UNUSED MPI_Comm comm, const std::string& path) override
    {
        const auto fname = createCheckpointName(path, "MembraneInt", "txt");
        const bool good = text_IO::read(fname, stepGen_);
        if (!good) die("failed to read '%s'\n", fname.c_str());
    }

    /// get the name of the type
    static std::string getTypeName()
    {
        return constructTypeName<TriangleInteraction, DihedralInteraction, Filter>(
                "MembraneInteraction");
    }

    void saveSnapshotAndRegister(Saver& saver) override
    {
        saver.registerObject<MembraneInteraction>(this, _saveSnapshot(saver, getTypeName()));
    }

protected:
    /** \brief Implementation of snapshot saving. Reusable by potential derived classes.
        \param [in,out] saver The \c Saver object. Provides save context and serialization functions.
        \param [in] typeName The name of the type being saved.
    */
    ConfigObject _saveSnapshot(Saver& saver, const std::string& typeName)
    {
        ConfigObject config = BaseMembraneInteraction::_saveSnapshot(saver, typeName);
        config.emplace("initLengthFraction",       saver(initLengthFraction_));
        config.emplace("growUntil",      saver(growUntil_));
        config.emplace("parameters",     saver(parameters_));
        config.emplace("dihedralParams", saver(dihedralParams_));
        config.emplace("triangleParams", saver(triangleParams_));
        config.emplace("filter",         saver(filter_));
        config.emplace("stepGen",        saver("<<not implemented>>"));
        return config;
    }

private:
    void _precomputeQuantities(MembraneVector *mv, cudaStream_t stream) override
    {
        BaseMembraneInteraction::_precomputeQuantities(mv, stream);

        precomputeQuantitiesPerEnergy(dihedralParams_, mv, stream);
        precomputeQuantitiesPerEnergy(triangleParams_, mv, stream);
    }

    /** Get a scaling factor for transforming the length scale of all the parameters
        (see also rescaleParameters())
        \param [in] t The simulation time (must be positive)
        \return scaling factor for length

        Will grow linearly from initLengthFraction_ to 1 during the first growUntil_ time interval.
        Otherwise this returns 1.
     */
    real _scaleFromTime(real t) const
    {
        return math::min(1.0_r, initLengthFraction_ + (1.0_r - initLengthFraction_) * (t / growUntil_));
    }

private:

    real initLengthFraction_; ///< Initial length scale (fraction); See _scaleFromTime()
    real growUntil_;          ///< Time required o recover the full size; See _scaleFromTime()
    CommonMembraneParameters parameters_; ///< parameters common to all variants of this object
    typename DihedralInteraction::ParametersType dihedralParams_; ///< dihedral forces parameters
    typename TriangleInteraction::ParametersType triangleParams_; ///< traingle forces parameters
    Filter filter_; ///< describes the cells to apply the forces to
    StepRandomGen stepGen_; ///< RNG
};

} // namespace mirheo
