#pragma once

#include "core/mirheo_object.h"

#include <cuda_runtime.h>
#include <mpi.h>
#include <string>
#include <vector>

class CellList;
class ParticleVector;
class ObjectVector;

/**
 * Interface for a class implementing bouncing from objects
 */
class Bouncer : public MirSimulationObject
{
public:
    Bouncer(const MirState *state, std::string name);
    virtual ~Bouncer();

    /**
     * Second step of initialization, called from the \c Simulation
     * All the preparation for bouncing must be done here
     */
    virtual void setup(ObjectVector *ov);

    /**
     * return which object vector the bounce is performed against
     */
    ObjectVector* getObjectVector();
    
    /**
     * Ask \c ParticleVector which the class will be working with to have specific properties
     * Default: ask nothing
     * Called from \c Simulation right after setup
     */
    virtual void setPrerequisites(ParticleVector *pv);

    /// Interface to the private exec function for local objects
    void bounceLocal(ParticleVector *pv, CellList *cl, cudaStream_t stream);

    /// Interface to the private exec function for halo objects
    void bounceHalo (ParticleVector *pv, CellList *cl, cudaStream_t stream);

    /// return list of extra channel names to be exchanged
    virtual std::vector<std::string> getChannelsToBeExchanged() const = 0;

protected:
    ObjectVector *ov;  /// Particles will be bounced against that ObjectVector

    /**
     * Should be defined to implement bounce.
     * Will be called from \c Simulation after the integration is done
     * and the objects are exchanged
     *
     * @param pv ptr to \c ParticleVector whose particles will be
     * bounced from the objects associated with this bouncer
     * @param cl ptr to \c CellList that has to be built for \c pv
     * @param local if \c true, will bounce from the local objects, if \c false -- from halo objects.
     *
     * \rst
     * .. note::
     *    Particles from \c pv (that actually will be bounced back) are always local
     * \endrst
     * @param stream cuda stream on which to execute
     */
    virtual void exec (ParticleVector *pv, CellList *cl, bool local, cudaStream_t stream) = 0;
};
