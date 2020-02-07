#pragma once

#include <cuda_runtime.h>
#include <mpi.h>

#include <mirheo/core/mirheo_object.h>

namespace mirheo
{

class ParticleVector;


/**
   \defgroup  Integrators Integrators
 */

/** \brief Advance \c ParticleVector objects in time
    \ingroup Integrators

    \c Integrator objects are responsible to advance the state of \c ParticleVector 
    objects on the device.
    After executed, the \c CellList of the \c ParticleVector object might be outdated;
    in that case, the \c Integrator is invalidates the current cell-lists, halo and 
    redistribution status on the \c ParticleVector. 
 */
class Integrator : public MirSimulationObject
{
public:
    
    /// Set the name of the integrator and state
    Integrator(const MirState *state, const std::string& name);

    virtual ~Integrator();

    virtual void execute(ParticleVector *pv, cudaStream_t stream) = 0;

    /**
     * Ask ParticleVectors which the class will be working with to have specific properties
     * Default: ask nothing
     * Called from Simulation right after setup
     */
    virtual void setPrerequisites(ParticleVector *pv);

protected:
    void invalidatePV_(ParticleVector *pv);
};

} // namespace mirheo
