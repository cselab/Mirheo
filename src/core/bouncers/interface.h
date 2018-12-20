#pragma once

#include <cuda_runtime.h>
#include <mpi.h>

#include "core/ymero_object.h"

class CellList;
class ParticleVector;
class ObjectVector;


/**
 * Interface for a class implementing bouncing from objects
 */
class Bouncer : public YmrSimulationObject
{
public:
    Bouncer(std::string name, const YmrState *state) :
        YmrSimulationObject(name, state)
    {}

    /**
     * Second step of initialization, called from the \c Simulation
     * All the preparation for bouncing must be done here
     */
    virtual void setup(ObjectVector* ov) = 0;

    /**
     * Ask \c ParticleVector which the class will be working with to have specific properties
     * Default: ask nothing
     * Called from \c Simulation right after setup
     */
    virtual void setPrerequisites(ParticleVector* pv) {}

    /// Interface to the private exec function for local objects
    void bounceLocal(ParticleVector* pv, CellList* cl, float dt, cudaStream_t stream) { exec (pv, cl, dt, true,  stream); }

    /// Interface to the private exec function for halo objects
    void bounceHalo (ParticleVector* pv, CellList* cl, float dt, cudaStream_t stream) { exec (pv, cl, dt, false, stream); }

protected:
    ObjectVector* ov;  /// Particles will be bounced against that ObjectVector

    /**
     * Should be defined to implement bounce.
     * Will be called from \c Simulation after the integration is done
     * and the objects are exchanged
     *
     * @param pv ptr to \c ParticleVector whose particles will be
     * bounced from the objects associated with this bouncer
     * @param cl ptr to \c CellList that has to be built for \c pv
     * @param dt timestep used to integrate \c pv
     * @param local if \c true, will bounce from the local objects, if \c false -- from halo objects.
     *
     * \rst
     * .. note::
     *    Particles from \c pv (that actually will be bounced back) are always local
     * \endrst
     * @param stream cuda stream on which to execute
     */
    virtual void exec (ParticleVector* pv, CellList* cl, float dt, bool local, cudaStream_t stream) = 0;
};
