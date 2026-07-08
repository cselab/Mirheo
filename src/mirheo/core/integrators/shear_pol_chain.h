// Copyright 2022 ETH Zurich. All Rights Reserved.
#pragma once

#include "interface.h"

#include <mirheo/core/utils/macros.h>

#include <array>

namespace mirheo {

/** \brief Set ParticleVector velocities a linear shear.

    The positions are integrated with forwards Euler.
    Evolve the polymeric chain vectors with forwards Euler.
 */
class IntegratorShearPolChain : public Integrator
{
public:
    /** \param [in] state The global state of the system. The time step and domain used during the execution are passed through this object.
        \param [in] name The name of the integrator.
        \param [in] origin a point with zero flow (in global coordinates)
        \param [in] shear shear rate tensor.
    */
    IntegratorShearPolChain(const MirState *state, const std::string& name,
                    std::array<real,9> shear, real3 origin);

    void execute(ParticleVector *pv, cudaStream_t stream) override;

private:
    std::array<real,9> shear_;   ///< Shear rate tensor
    real3 origin_;  ///< Point with zero flow
};

} // namespace mirheo
