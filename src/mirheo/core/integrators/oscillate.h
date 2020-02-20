#pragma once

#include "interface.h"

#include <mirheo/core/utils/macros.h>

namespace mirheo
{

/** \brief Restrict ParticleVector velocities to a sine wave.
    
    \rst
    Set velocities to follow a sine wave:
    
    .. math::
        v(t) = v \cos\left(\frac{ 2 \pi t} {T}\right)

    The positions are integrated with forwards euler from the above velocities.
    \endrst
 */
class IntegratorOscillate : public Integrator
{
public:
    /** \param [in] state The global state of the system. The time step and domain used during the execution are passed through this object.
        \param [in] name The name of the integrator.
        \param [in] vel Velocity magnitude.
        \param [in] period The time taken for one oscillation.
    */
    IntegratorOscillate(const MirState *state, const std::string& name, real3 vel, real period);
    ~IntegratorOscillate();

    void execute(ParticleVector *pv, cudaStream_t stream) override;

private:
    real3 vel_;    ///< Velocity amplitude
    real period_;  ///< Sine wave period
};

} // namespace mirheo
