#pragma once

#include "interface.h"

#include <mirheo/core/containers.h>
#include <mirheo/core/datatypes.h>

#include <vector>

namespace mirheo
{

class Interaction;

/** \brief Advance one \c ObjectVector associated with internal forces with smaller time step.
    \ingroup Integrators

    We distinguish slow forces, which are computed outside of this class, from fast forces, 
    computed only inside this class.
    Each time step given by the simulation is split into n sub time steps.
    Each of these sub time step advances the object using the non updated slow forces and the updated 
    fast forces n times.

    This was motivated by the separation of time scale of membrane viscosity (fast forces) and solvent
    viscosity (slow forces) in blood.

    \rst
    .. warning::
        The fast forces should NOT be registered in the \c Simulation.
        Otherwise, it will be executed twice (by the simulation and by this class).
    \endrst
 */
class IntegratorSubStep : public Integrator
{
public:
    /** \brief construct a \c IntegratorSubStep object.
        \param [in] state The global state of the system. The time step and domain used during the execution are passed through this object.
        \param [in] name The name of the integrator.
        \param [in] substeps Number of sub steps
        \param [in] fastForces Internal interactions executed at each sub step.

        This constructor will die if the fast forces need to exchange ghost particles with other ranks.
    */
    IntegratorSubStep(const MirState *state, const std::string& name, int substeps,
                      const std::vector<Interaction*>& fastForces);
    ~IntegratorSubStep();
    
    void execute(ParticleVector *pv, cudaStream_t stream) override;

    void setPrerequisites(ParticleVector *pv) override;

private:

    std::vector<Interaction*> fastForces_; /* interactions (self) called `substeps` times per time step */
    std::unique_ptr<Integrator> subIntegrator_;
    MirState subState_;
    
    int substeps_; /* number of substeps */
    DeviceBuffer<Force> slowForces_ {};
    DeviceBuffer<real4> previousPositions_ {};

    void updateSubState_();
};

} // namespace mirheo
