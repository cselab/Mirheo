#include "from_function.h"

#include <core/utils/cuda_common.h>

FieldFromFunction::FieldFromFunction(const MirState *state, std::string name, FieldFunction func, float3 h) :
    Field(state, name, h),
    func(func)
{}

FieldFromFunction::~FieldFromFunction() = default;

FieldFromFunction::FieldFromFunction(FieldFromFunction&&) = default;


inline float make_perioidc(float x, float L)
{
    if (x <  0) x += L;
    if (x >= L) x -= L;
    return x;
};

inline float3 make_periodic(float3 r, float3 L)
{
    return {make_perioidc(r.x, L.x),
            make_perioidc(r.y, L.y),
            make_perioidc(r.z, L.z)};
}

void FieldFromFunction::setup(const MPI_Comm& comm)
{
    info("Setting up field '%s'", name.c_str());

    const auto domain = state->domain;
    
    CUDA_Check( cudaDeviceSynchronize() );
    
    PinnedBuffer<float> fieldRawData (resolution.x * resolution.y * resolution.z);

    int3 i;
    int id = 0;
    for (i.z = 0; i.z < resolution.z; ++i.z) {
        for (i.y = 0; i.y < resolution.y; ++i.y) {
            for (i.x = 0; i.x < resolution.x; ++i.x) {
                float3 r {i.x * h.x, i.y * h.y, i.z * h.z};
                r -= extendedDomainSize*0.5f;
                r  = domain.local2global(r);
                r  = make_periodic(r, domain.globalSize);
                
                fieldRawData[id++] = func(r);
            }
        }
    }

    fieldRawData.uploadToDevice(defaultStream);
    
    setupArrayTexture(fieldRawData.devPtr());
}
