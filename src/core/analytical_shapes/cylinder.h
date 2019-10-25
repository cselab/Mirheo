#pragma once

#include <core/utils/cpu_gpu_defines.h>
#include <core/utils/cuda_common.h>
#include <core/utils/helper_math.h>

class Cylinder
{
public:
    Cylinder(real R, real L) :
        R(R),
        halfL(0.5f * L)
    {}

    __HD__ inline real inOutFunction(real3 coo) const
    {
        const real dr = math::sqrt(sqr(coo.x) + sqr(coo.y)) - R;
        const real dz = math::abs(coo.z) - halfL;

        const real dist2edge = math::sqrt(sqr(dz) + sqr(dr));
        const real dist2disk = dr > 0 ? dist2edge : dz;
        const real dist2cyl  = dz > 0 ? dist2edge : dr;

        return (dz <= 0) && (dr <= 0)
            ? math::max(dist2cyl, dist2disk)
            : math::min(dist2cyl, dist2disk);
    }

    __HD__ inline real3 normal(real3 coo) const
    {
        constexpr real eps   = 1e-6f;
        constexpr real delta = 1e-3f;
        
        const real rsq = sqr(coo.x) + sqr(coo.y);
        const real rinv = rsq > eps ? math::rsqrt(rsq) : 0.f;

        const real dr = math::sqrt(rsq) - R;
        const real dz = math::abs(coo.z) - halfL;
        
        const real3 er {rinv * coo.x, rinv * coo.y, 0.f};
        const real3 ez {0.f, 0.f, coo.z > 0 ? 1.f : -1.f};

        
        real3 n {0.f, 0.f, 0.f};
        if (math::abs(dr) < delta) n += er;
        if (math::abs(dz) < delta) n += ez;
        return n;
    }
    

    inline real3 inertiaTensor(real totalMass) const
    {
        const real xx = totalMass * R * R * 0.25f;
        const real yy = xx;
        const real zz = totalMass * halfL * halfL * 0.3333333f;
        
        return make_real3(yy + zz, xx + zz, xx + yy);
    }

    static const char *desc;
    
private:
    
    real R, halfL;
};
