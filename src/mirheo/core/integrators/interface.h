// Copyright 2020 ETH Zurich. All Rights Reserved.
#pragma once

#include <cuda_runtime.h>
#include <mpi.h>

#include <mirheo/core/mirheo_object.h>

namespace mirheo
{

class ParticleVector;

/** \brief Advance ParticleVector objects in time.

    \c Integrator objects are responsible to advance the state of ParticleVector
    objects on the device.
    After executed, the CellList of the ParticleVector object might be outdated;
    in that case, the \c Integrator is invalidates the current cell-lists, halo and
    redistribution status on the ParticleVector.
 */
class Integrator : public MirSimulationObject
{
public:

    /** \brief Construct a \c Integrator object.
        \param [in] state The global state of the system. The time step and domain used during the execution are passed through this object.
        \param [in] name The name of the integrator.
    */
    Integrator(const MirState *state, const std::string& name);
    virtual ~Integrator();

    /** \brief Setup conditions on the ParticledVector.
        \param [in,out] pv The ParticleVector that will be advanced in time.

        Set specific properties to pv that will be modified during execute().
        Default: ask nothing.
        Must be called before execute() with the same pv.
     */
    virtual void setPrerequisites(ParticleVector *pv);


    /** \brief Advance the ParticledVector for one time step.
        \param [in,out] pv The ParticleVector that will be advanced in time.
        \param [in] stream The stream used for execution.
     */
    virtual void execute(ParticleVector *pv, cudaStream_t stream) = 0;

protected:
    /** \brief Invalidate ParticledVector cell-lists, halo and redistributed statuses.
        \param [in,out] pv The ParticleVector that must be invalidated.
     */
    void invalidatePV_(ParticleVector *pv);

    /** \brief Snapshot base function. Sets the category to "Integrator".
        \param [in,out] saver The \c Saver object. Provides save context and serialization functions.
        \param [in] typeName The name of the type being saved, the `__type` field.
     */
    ConfigObject _saveSnapshot(Saver& saver, const std::string& typeName);
};

} // namespace mirheo
