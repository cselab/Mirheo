#pragma once

#include <mirheo/core/domain.h>
#include <mirheo/core/datatypes.h>

#include <mirheo/core/utils/cpu_gpu_defines.h>
#include <mirheo/core/utils/helper_math.h>

namespace mirheo
{

class ParticleVector;

class VelocityFieldOscillate
{
public:
    VelocityFieldOscillate(real3 vel, real period) :
        vel_(vel),
        period_(period)
    {
        if (period_ <= 0)
            die("Oscillating period should be strictly positive");
    }

    void setup(real t, __UNUSED DomainInfo domain)
    {
        cosOmega_ = math::cos(2*M_PI * t / period_);
    }

    const VelocityFieldOscillate& handler() const { return *this; }

    __D__ inline real3 operator()(__UNUSED real3 coo) const
    {
        return vel_ * cosOmega_;
    }

private:
    real3 vel_;
    real period_;
    real cosOmega_ {0.0_r};
};

} // namespace mirheo
