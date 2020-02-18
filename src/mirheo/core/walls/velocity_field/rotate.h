#pragma once

#include <mirheo/core/domain.h>
#include <mirheo/core/datatypes.h>

#include <mirheo/core/utils/cpu_gpu_defines.h>
#include <mirheo/core/utils/helper_math.h>

namespace mirheo
{

class ParticleVector;

/** Rotating velocity field (constant in time).
    The field is defined by a center and an angular velocity:

    \rst
    .. math::
        \mathbf{v}(\mathbf{r}) = \mathbf{\omega} \times (\mathbf{r} - \mathbf{c})
    \endrst
*/
class VelocityFieldRotate
{
public:
    /** Construct a \c VelocityFieldRotate object
        \param [in] omega The angular velocity
        \param [in] center Center of rotation in global coordinates
    */
    VelocityFieldRotate(real3 omega, real3 center) :
        omega_(omega),
        center_(center)
    {}

    /** Synchronize with simulation state. Must be called at every time step.
        \param [in] t Simulation time. 
        \param [in] domain domain info.
     */
    void setup(__UNUSED real t, DomainInfo domain)
    {
        domain_ = domain;
    }

    /// get a handler that can be used on the device.
    const VelocityFieldRotate& handler() const
    {
        return *this;
    }

    /** Evaluate the velocity field at a given position
        \param [in] r The position in local coordinates
        \return The velocity value
    */
    __D__ inline real3 operator()(real3 r) const
    {
        const real3 gr = domain_.local2global(r);
        return cross(omega_, gr - center_);
    }

private:
    real3 omega_;
    real3 center_;

    DomainInfo domain_;
};

} // namespace mirheo
