#pragma once

#include "interface.h"

#include <mirheo/core/utils/macros.h>

namespace mirheo
{

/** \brief Restrict ParticleVector velocities to a constant.
    
    The positions are integrated with forwards euler with a constant velocity.
 */
class IntegratorTranslate : public Integrator
{
public:
    /** \param [in] state The global state of the system. The time step and domain used during the execution are passed through this object.
        \param [in] name The name of the integrator.
        \param [in] vel Velocity magnitude.
    */
    IntegratorTranslate(const MirState *state, const std::string& name, real3 vel);
    ~IntegratorTranslate();

    void execute(ParticleVector *pv, cudaStream_t stream) override;

private:
    real3 vel_;   ///< Velocity
};

} // namespace mirheo
