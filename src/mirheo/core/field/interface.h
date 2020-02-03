#pragma once

#include <mirheo/core/domain.h>
#include <mirheo/core/containers.h>
#include <mirheo/core/mirheo_object.h>

#include <mirheo/core/utils/cpu_gpu_defines.h>
#include <mirheo/core/utils/helper_math.h>

#include <vector>

namespace mirheo
{

#ifndef __NVCC__
template<typename T>
T tex3D(__UNUSED cudaTextureObject_t t,
        __UNUSED real x, __UNUSED real y, __UNUSED real z)
{
    return T();
}
#endif

class FieldDeviceHandler
{
public:
    __D__ inline real operator()(real3 x) const
    {
        //https://en.wikipedia.org/wiki/Trilinear_interpolation
        real s000, s001, s010, s011, s100, s101, s110, s111;
        real sx00, sx01, sx10, sx11, sxy0, sxy1, sxyz;

        const real3 texcoord = math::floor((x + extendedDomainSize_*0.5_r) * invh_);
        const real3 lambda = (x - (texcoord * h_ - extendedDomainSize_*0.5_r)) * invh_;
        
        auto access = [this, &texcoord] (int dx, int dy, int dz)
        {
            const auto val = tex3D<float>(fieldTex_,
                                          static_cast<float>(texcoord.x + static_cast<real>(dx)),
                                          static_cast<float>(texcoord.y + static_cast<real>(dy)),
                                          static_cast<float>(texcoord.z + static_cast<real>(dz)));
            return static_cast<real>(val);
        };
        
        s000 = access(0, 0, 0);
        s001 = access(0, 0, 1);
        s010 = access(0, 1, 0);
        s011 = access(0, 1, 1);
        
        s100 = access(1, 0, 0);
        s101 = access(1, 0, 1);
        s110 = access(1, 1, 0);
        s111 = access(1, 1, 1);
        
        sx00 = s000 * (1 - lambda.x) + lambda.x * s100;
        sx01 = s001 * (1 - lambda.x) + lambda.x * s101;
        sx10 = s010 * (1 - lambda.x) + lambda.x * s110;
        sx11 = s011 * (1 - lambda.x) + lambda.x * s111;

        sxy0 = sx00 * (1 - lambda.y) + lambda.y * sx10;
        sxy1 = sx01 * (1 - lambda.y) + lambda.y * sx11;

        sxyz = sxy0 * (1 - lambda.z) + lambda.z * sxy1;

        return sxyz;
    }

protected:

    cudaTextureObject_t fieldTex_;
    real3 h_, invh_, extendedDomainSize_;
};


class Field : public FieldDeviceHandler, public MirSimulationObject
{
public:    
    Field(const MirState *state, std::string name, real3 h);
    virtual ~Field();

    Field(Field&&);
    
    const FieldDeviceHandler& handler() const;

    virtual void setup(const MPI_Comm& comm) = 0;
    
protected:

    int3 resolution_;
    
    cudaArray *fieldArray_;
    
    const real3 margin3_{5, 5, 5};

    void setupArrayTexture(const float *fieldDevPtr);
};

} // namespace mirheo
