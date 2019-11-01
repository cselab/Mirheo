#pragma once

#include <cuda_runtime.h>
#include <mpi.h>

#include <mirheo/core/mirheo_object.h>

namespace mirheo
{

class ParticleVector;

/**
 * Integrate ParticleVectors
 *
 * Should implement movement of the particles or objects due to the applied forces.
 *
 * \rst
 * .. attention::
 *    The interface defines two integration stages that should be called before
 *    and after the forces are computed, but currently stage1() will never be called.
 *    So all the integration for now is done in stage2()
 * \endrst
 */
class Integrator : public MirSimulationObject
{
public:
    
    /// Set the name of the integrator and state
    Integrator(const MirState *state, std::string name);

    virtual ~Integrator();

    /**
     * First integration stage, to be called before the forces are computed
     *
     * @param pv ParticleVector to be integrated
     */
    virtual void stage1(ParticleVector *pv, cudaStream_t stream) = 0;

    /**
     * Second integration stage, to be called after the forces are computed
     *
     * @param pv ParticleVector to be integrated
     */
    virtual void stage2(ParticleVector *pv, cudaStream_t stream) = 0;

    /**
     * Ask ParticleVectors which the class will be working with to have specific properties
     * Default: ask nothing
     * Called from Simulation right after setup
     */
    virtual void setPrerequisites(ParticleVector *pv);

protected:
    void invalidatePV(ParticleVector *pv);
};

} // namespace mirheo
