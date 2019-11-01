#pragma once

#include <core/utils/cpu_gpu_defines.h>
#include <core/utils/cuda_common.h>
#include <core/utils/helper_math.h>

class Cylinder
{
public:
    Cylinder(real R, real L) :
        R(R),
        halfL(0.5_r * L)
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
        constexpr real eps   = 1e-6_r;
        constexpr real delta = 1e-3_r;
        
        const real rsq = sqr(coo.x) + sqr(coo.y);
        const real rinv = rsq > eps ? math::rsqrt(rsq) : 0._r;

        const real dr = math::sqrt(rsq) - R;
        const real dz = math::abs(coo.z) - halfL;
        
        const real3 er {rinv * coo.x, rinv * coo.y, 0._r};
        const real3 ez {0._r, 0._r, coo.z > 0 ? 1._r : -1._r};

        
        real3 n {0._r, 0._r, 0._r};
        if (math::abs(dr) < delta) n += er;
        if (math::abs(dz) < delta) n += ez;
        return n;
    }
    

    inline real3 inertiaTensor(real totalMass) const
    {
        const real xx = totalMass * R * R * 0.25_r;
        const real yy = xx;
        const real zz = totalMass * halfL * halfL * 0.3333333_r;
        
        return make_real3(yy + zz, xx + zz, xx + yy);
    }

    static const char *desc;
    
private:
    
    real R, halfL;
};
