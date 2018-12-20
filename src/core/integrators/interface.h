#pragma once

#include <cuda_runtime.h>
#include <mpi.h>

#include "core/ymero_object.h"

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
class Integrator : public YmrSimulationObject
{
public:
    
    /// Set the name of the integrator and state
    Integrator(std::string name, const YmrState *state) :
        YmrSimulationObject(name, state),
        dt(state->dt)
    {}

    /**
     * First integration stage, to be called before the forces are computed
     *
     * @param pv ParticleVector to be integrated
     * @param t current simulation time
     */
    virtual void stage1(ParticleVector* pv, float t, cudaStream_t stream) = 0;

    /**
     * Second integration stage, to be called after the forces are computed
     *
     * @param pv ParticleVector to be integrated
     * @param t current simulation time
     */
    virtual void stage2(ParticleVector* pv, float t, cudaStream_t stream) = 0;

    /**
     * Ask ParticleVectors which the class will be working with to have specific properties
     * Default: ask nothing
     * Called from Simulation right after setup
     */
    virtual void setPrerequisites(ParticleVector* pv) {}

public:
    float dt; /// allow to get different timestep than global timestep found in state
};
