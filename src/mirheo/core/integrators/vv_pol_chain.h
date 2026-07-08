// Copyright 2022 ETH Zurich. All Rights Reserved.
#pragma once

#include "interface.h"

namespace mirheo {

/** \brief Advance individual particles with Velocity-Verlet scheme, and evolve their polymeric chain vector.
 */
class IntegratorVVPolChain : public Integrator
{
public:
    /** \param [in] state The global state of the system.
        \param [in] name The name of the integrator.
    */
    IntegratorVVPolChain(const MirState *state, const std::string& name);

    void execute(ParticleVector *pv, cudaStream_t stream) override;
};

} // namespace mirheo
