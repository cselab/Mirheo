#pragma once

#include <mirheo/core/datatypes.h>
#include <mirheo/core/utils/cpu_gpu_defines.h>

namespace mirheo
{

/** \brief Base class for analytic shape representation

    Anlytic shapes are used to represent the surface of objectsin their 
    local frame of reference. 
    The represntation of that surface is implicit (see inOutFunction()).
    These shape representations are used inside e.g. in \c ObjectBelongingChecker 
    and \c Bouncer.

    The shapes must be oriented such as their inertia tensor is diagonal (see inertiaTensor()).

    Every newly added analytic shape must derive from that base class. 
    Furthermore, an entry should be added in the xmacro in api.h.
 */
class AnalyticShape
{
public:
    /**\brief Implicit surface representation.
       \param [in] r The position at which to evaluate the in/out field.
       \return The value of the field at the given position.

       This scalar field is a smooth function of the position.
       It is negative inside the object represented by the surface and positive outside.
       The zero level set of that field must coincides with the surface.
    */
    virtual __HD__ real inOutFunction(real3 r) const = 0;

    /**\brief Get the normal pointing outside the object.
       \param [in] r The position at which to evaluate the normal.
       \return The normal at r (length of this return must be 1).

       This vector field is a smooth function of the position.
       On the surface, it represents the normal vector of the surface.
    */
    virtual __HD__ real3 normal(real3 r) const = 0;


    /**\brief Get the inertia tensor of the shape in its frame of reference.
       \param [in] totalMass The total mass of the object.
       \return The diagonal of the inertia tensor. The object must be aligned 
               such that the inertia tensor is diagonal.
    */
    virtual real3 inertiaTensor(real totalMass) const = 0;
};

} // namespace mirheo
