// Copyright 2020 ETH Zurich. All Rights Reserved.
#pragma once

#include "interface.h"

#include <mirheo/core/containers.h>
#include <mirheo/core/datatypes.h>
#include <mirheo/core/mesh/edge_colors.h>

#include <map>
#include <memory>
#include <random>
#include <vector>

namespace mirheo
{

class BaseMembraneInteraction;
class MembraneVector;

/** \brief Advance one MembraneVector associated with internal forces with smaller time step, similar to IntgratorSubStep.

    We distinguish slow forces, which are computed outside of this class, from fast forces,
    computed only inside this class.
    Each time step given by the simulation is split into n sub time steps.
    Each of these sub time step advances the object using the non updated slow forces and the updated
    fast forces n times using the Shardlow method.

    This was motivated by the separation of time scale of membrane viscosity (fast forces) and solvent
    viscosity (slow forces) in blood.

    \rst
    .. warning::
        The fast forces should NOT be registered in the \c Simulation.
        Otherwise, it will be executed twice (by the simulation and by this class).
    \endrst
 */
class IntegratorSubStepShardlowSweep : public Integrator
{
public:
    /** \brief construct a IntegratorSubStepShardlowSweep object.
        \param [in] state The global state of the system. The time step and domain used during the execution are passed through this object.
        \param [in] name The name of the integrator.
        \param [in] substeps Number of sub steps
        \param [in] fastForces Internal interactions executed at each sub step.

    */
    IntegratorSubStepShardlowSweep(const MirState *state, const std::string& name, int substeps,
                                   BaseMembraneInteraction* fastForces, real gammaC, real kBT, int nsweeps);


    ~IntegratorSubStepShardlowSweep();

    void execute(ParticleVector *pv, cudaStream_t stream) override;

    void setPrerequisites(ParticleVector *pv) override;

private:
    void _viscousSweeps(MembraneVector *mv, cudaStream_t stream);

private:
    int substeps_; /* number of substeps */

    BaseMembraneInteraction *fastForces_; /* interactions (self) called `substeps` times per time step */
    real gammaC_;
    real kBT_;
    int nsweeps_;

    MirState subState_;

    DeviceBuffer<Force> slowForces_ {};
    DeviceBuffer<real4> previousPositions_ {};

    std::mt19937 rnd_;
    std::map<std::string, std::unique_ptr<MeshDistinctEdgeSets>> pvToEdgeSets_;
};

} // namespace mirheo
