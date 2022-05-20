// Copyright 2020 ETH Zurich. All Rights Reserved.
#pragma once

#include "interface.h"

namespace mirheo
{

/** \brief Energy minimization integrator.

     Updates positions using a force-based gradient descent, without affecting
     or reading velocities.
 */
class IntegratorMinimize : public Integrator
{
public:
    /** \param [in] state The global state of the system. The time step and domain used during the execution are passed through this object.
        \param [in] name The name of the integrator.
        \param [in] maxDisplacement Maximum particle displacement per time step.
    */
    IntegratorMinimize(const MirState *state, const std::string& name, real maxDisplacement);

    void execute(ParticleVector *pv, cudaStream_t stream) override;

private:
    real maxDisplacement_; ///< Maximum displacement per time step.
};

} // namespace mirheo
