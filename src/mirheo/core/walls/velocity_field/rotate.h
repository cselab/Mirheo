#pragma once

#include <mirheo/core/domain.h>
#include <mirheo/core/datatypes.h>

#include <mirheo/core/utils/cpu_gpu_defines.h>
#include <mirheo/core/utils/helper_math.h>

namespace mirheo
{

class ParticleVector;

class VelocityField_Rotate
{
public:
    VelocityField_Rotate(real3 omega, real3 center) :
        omega_(omega),
        center_(center)
    {}

    void setup(__UNUSED real t, DomainInfo domain)
    {
        domain_ = domain;
    }

    const VelocityField_Rotate& handler() const
    {
        return *this;
    }

    __D__ inline real3 operator()(real3 coo) const
    {
        const real3 gr = domain_.local2global(coo);
        return cross(omega_, gr - center_);
    }

private:
    real3 omega_;
    real3 center_;

    DomainInfo domain_;
};

} // namespace mirheo
