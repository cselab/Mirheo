#pragma once

#include <mirheo/core/domain.h>
#include <mirheo/core/datatypes.h>

#include <mirheo/core/utils/cpu_gpu_defines.h>
#include <mirheo/core/utils/helper_math.h>

#include <functional>

namespace mirheo
{

class ParticleVector;

using VelocityTimeFunction = std::function<real3(real)>;

/** Uniform velocity field changing over time
    provided by a lambda function.
*/
class VelocityFieldLambda
{
public:
    /** \brief Construct a VelocityFieldLambda object.
        \param [in] vfunc The velocity function of time
    */
    VelocityFieldLambda(VelocityTimeFunction vfunc) :
        vfunc_(std::move(vfunc))
    {}

    /** Synchronize with simulation state. Must be called at every time step.
        \param [in] t Simulation time.
        \param [in] domain domain info.
     */
    void setup(real t, __UNUSED DomainInfo domain)
    {
        vel_ = vfunc_(t);
    }

    /// get a handler that can be used on the device.
    const VelocityFieldLambda& handler() const { return *this; }

    /** Evaluate the velocity field at a given position
        \param [in] r The position in local coordinates
        \return The velocity value
    */
    __D__ inline real3 operator()(__UNUSED real3 r) const
    {
        return vel_;
    }

private:
    VelocityTimeFunction vfunc_;
    real3 vel_ {0.0_r, 0.0_r, 0.0_r};
};

} // namespace mirheo
