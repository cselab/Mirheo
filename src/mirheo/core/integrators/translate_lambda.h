// Copyright 2020 ETH Zurich. All Rights Reserved.
#pragma once

#include "interface.h"

#include <mirheo/core/utils/macros.h>

#include <functional>

namespace mirheo
{

/** \brief Restrict ParticleVector velocities to a uniform function of time.

    \rst
    Set velocities to follow a given function of time.
    Positions are integrated with forwards euler.
    \endrst
 */
class IntegratorTranslateLambda : public Integrator
{
public:
    /** \param [in] state The global state of the system. The time step and domain used during the execution are passed through this object.
        \param [in] name The name of the integrator.
        \param [in] vel Velocity function of time.
    */
    IntegratorTranslateLambda(const MirState *state, const std::string& name, std::function<real3(real)> vel);
    ~IntegratorTranslateLambda();

    void execute(ParticleVector *pv, cudaStream_t stream) override;

private:
    std::function<real3(real)> vel_; ///< Velocity
};

} // namespace mirheo
