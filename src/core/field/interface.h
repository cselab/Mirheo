#pragma once

#include <core/domain.h>
#include <core/containers.h>

#include <core/utils/cpu_gpu_defines.h>
#include <core/utils/helper_math.h>

#include <vector>

#ifndef __NVCC__
template<typename T>
T tex3D(cudaTextureObject_t t, float x, float y, float z)
{
    return 0.0f;
}
#endif

class FieldDeviceHandler
{
public:
    __D__ inline float operator()(float3 x) const
    {
        //https://en.wikipedia.org/wiki/Trilinear_interpolation
        float s000, s001, s010, s011, s100, s101, s110, s111;
        float sx00, sx01, sx10, sx11, sxy0, sxy1, sxyz;

        float3 texcoord = floorf((x + extendedDomainSize*0.5f) * invh);
        float3 lambda = (x - (texcoord * h - extendedDomainSize*0.5f)) * invh;
        
        auto access = [this, &texcoord] (int dx, int dy, int dz) {
            return tex3D<float>(fieldTex, texcoord.x + dx, texcoord.y + dy, texcoord.z + dz);
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

    cudaTextureObject_t fieldTex;
    float3 h, invh, extendedDomainSize;    
};


class Field : public FieldDeviceHandler
{
public:    
    Field(std::string fieldFileName, float3 h);    
    ~Field();

    Field(Field&&);
    
    const FieldDeviceHandler& handler() const;

    void setup(MPI_Comm& comm, DomainInfo domain);
    
protected:

    int3 resolution;
    
    cudaArray *fieldArray;
    
    const float3 margin3{5, 5, 5};

    void setupArrayTexture(const float *fieldDevPtr);
    
    std::string fieldFileName;

    void prepareRelevantSdfPiece(int rank, const float* fullSdfData, float3 extendedDomainStart, float3 initialSdfH, int3 initialSdfResolution,
            int3& resolution, float3& offset, PinnedBuffer<float>& localSdfData);
};
