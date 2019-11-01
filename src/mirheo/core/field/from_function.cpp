#include "from_function.h"

#include <mirheo/core/utils/cuda_common.h>

namespace mirheo
{

FieldFromFunction::FieldFromFunction(const MirState *state, std::string name, FieldFunction func, real3 h) :
    Field(state, name, h),
    func(func)
{}

FieldFromFunction::~FieldFromFunction() = default;

FieldFromFunction::FieldFromFunction(FieldFromFunction&&) = default;


inline real make_perioidc(real x, real L)
{
    if (x <  0) x += L;
    if (x >= L) x -= L;
    return x;
}

inline real3 make_periodic(real3 r, real3 L)
{
    return {make_perioidc(r.x, L.x),
            make_perioidc(r.y, L.y),
            make_perioidc(r.z, L.z)};
}

void FieldFromFunction::setup(__UNUSED const MPI_Comm& comm)
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
                real3 r {i.x * h.x, i.y * h.y, i.z * h.z};
                r -= extendedDomainSize * 0.5_r;
                r  = domain.local2global(r);
                r  = make_periodic(r, domain.globalSize);
                
                fieldRawData[id++] = static_cast<float>(func(r));
            }
        }
    }

    fieldRawData.uploadToDevice(defaultStream);
    
    setupArrayTexture(fieldRawData.devPtr());
}

} // namespace mirheo
