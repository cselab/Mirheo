#include "from_function.h"

FieldFromFunction::FieldFromFunction(const YmrState *state, FieldFunction func, float3 h) :
    Field(state, h),
    func(func)
{}

FieldFromFunction::~FieldFromFunction() = default;

FieldFromFunction::FieldFromFunction(FieldFromFunction&&) = default;

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
                float3 x {i.x * h.x, i.y * h.y, i.z * h.z};
                x -= extendedDomainSize*0.5f;
                x = domain.local2global(x);
                
                fieldRawData[id++] = func(x);
            }
        }
    }

    fieldRawData.uploadToDevice(0);
    
    setupArrayTexture(fieldRawData.devPtr());
}
