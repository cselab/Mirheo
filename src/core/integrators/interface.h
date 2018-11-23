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
    float dt;

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

    /// Set the name of the integrator and its time-step
    Integrator(std::string name, float dt) : YmrSimulationObject(name), dt(dt) {}
};
