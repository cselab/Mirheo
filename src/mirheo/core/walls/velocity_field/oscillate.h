#pragma once

#include <mirheo/core/domain.h>
#include <mirheo/core/datatypes.h>

#include <mirheo/core/utils/cpu_gpu_defines.h>
#include <mirheo/core/utils/helper_math.h>

namespace mirheo
{

class ParticleVector;

/** Oscillating velocity field in time

    \rst
    .. math::
        \mathbf{v}(t) = \cos{\frac {2 \pi t}{T}} \mathbf{v},

    where :math:`T` is the period.
    \endrst
*/
class VelocityFieldOscillate
{
public:
    /** \brief Construct a \c VelocityFieldOscillate object.
        \param [in] vel The maximum velocity vector
        \param [in] period Oscillating period in simulation time. Fails if negative.
    */
    VelocityFieldOscillate(real3 vel, real period) :
        vel_(vel),
        period_(period)
    {
        if (period_ <= 0)
            die("Oscillating period should be strictly positive");
    }

    /** Synchronize with simulation state. Must be called at every time step.
        \param [in] t Simulation time. 
        \param [in] domain domain info.
     */
    void setup(real t, __UNUSED DomainInfo domain)
    {
        cosOmega_ = math::cos(2*M_PI * t / period_);
    }

    /// get a handler that can be used on the device.
    const VelocityFieldOscillate& handler() const { return *this; }
    
    /** Evaluate the velocity field at a given position
        \param [in] r The position in local coordinates
        \return The velocity value
    */
    __D__ inline real3 operator()(__UNUSED real3 r) const
    {
        return vel_ * cosOmega_;
    }

private:
    real3 vel_;
    real period_;
    real cosOmega_ {0.0_r};
};

} // namespace mirheo
