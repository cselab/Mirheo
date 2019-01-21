#include "from_function.h"

FieldFromFunction::FieldFromFunction(const YmrState *state, FieldFunction func, float3 h) :
    Field(state, h),
    func(func)
{}

FieldFromFunction::~FieldFromFunction() = default;

FieldFromFunction::FieldFromFunction(FieldFromFunction&&) = default;

static float3 make_periodic(float3 r, float3 L)
{
    auto oneD = [](float x, float L) {
        if (x <  0) x += L;
        if (x >= L) x -= L;
        return x;
    };

    return {oneD(r.x, L.x),
            oneD(r.y, L.y),
            oneD(r.z, L.z)};
}

void FieldFromFunction::setup(MPI_Comm& comm)
{
    info("Setting up field");

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

    fieldRawData.uploadToDevice(0);
    
    setupArrayTexture(fieldRawData.devPtr());
}
