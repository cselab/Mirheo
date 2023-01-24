// Copyright 2020 ETH Zurich. All Rights Reserved.
#include "from_function.h"

#include <mirheo/core/utils/cuda_common.h>

namespace mirheo {

ScalarFieldFromFunction::ScalarFieldFromFunction(const MirState *state, std::string name,
                                                 ScalarFieldFunction func, real3 h, real3 margin) :
    ScalarField(state, name, h, margin),
    func_(func)
{}

ScalarFieldFromFunction::~ScalarFieldFromFunction() = default;

ScalarFieldFromFunction::ScalarFieldFromFunction(ScalarFieldFromFunction&&) = default;


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

void ScalarFieldFromFunction::setup(__UNUSED const MPI_Comm& comm)
{
    info("Setting up field '%s'", getCName());

    const auto domain = getState()->domain;

    CUDA_Check( cudaDeviceSynchronize() );

    PinnedBuffer<float> fieldRawData (resolution_.x * resolution_.y * resolution_.z);

    int3 i;
    int id = 0;
    for (i.z = 0; i.z < resolution_.z; ++i.z) {
        for (i.y = 0; i.y < resolution_.y; ++i.y) {
            for (i.x = 0; i.x < resolution_.x; ++i.x) {
                real3 r {static_cast<real>(i.x) * h_.x,
                         static_cast<real>(i.y) * h_.y,
                         static_cast<real>(i.z) * h_.z};
                r -= extendedDomainSize_ * 0.5_r;
                r  = domain.local2global(r);
                r  = make_periodic(r, domain.globalSize);

                fieldRawData[id++] = static_cast<float>(func_(r));
            }
        }
    }

    fieldRawData.uploadToDevice(defaultStream);

    _setupArrayTexture(fieldRawData.devPtr());
}




VectorFieldFromFunction::VectorFieldFromFunction(const MirState *state, std::string name,
                                                 VectorFieldFunction func, real3 h, real3 margin) :
    VectorField(state, name, h, margin),
    func_(func)
{}

VectorFieldFromFunction::~VectorFieldFromFunction() = default;

VectorFieldFromFunction::VectorFieldFromFunction(VectorFieldFromFunction&&) = default;

void VectorFieldFromFunction::setup(__UNUSED const MPI_Comm& comm)
{
    info("Setting up field '%s'", getCName());

    const auto domain = getState()->domain;

    CUDA_Check( cudaDeviceSynchronize() );

    PinnedBuffer<float4> fieldRawData (resolution_.x * resolution_.y * resolution_.z);

    int3 i;
    int id = 0;
    for (i.z = 0; i.z < resolution_.z; ++i.z) {
        for (i.y = 0; i.y < resolution_.y; ++i.y) {
            for (i.x = 0; i.x < resolution_.x; ++i.x) {
                real3 r {static_cast<real>(i.x) * h_.x,
                         static_cast<real>(i.y) * h_.y,
                         static_cast<real>(i.z) * h_.z};
                r -= extendedDomainSize_ * 0.5_r;
                r  = domain.local2global(r);
                r  = make_periodic(r, domain.globalSize);

                const real3 val = func_(r);
                fieldRawData[id++] = make_float4(val.x, val.y, val.z, 0.0f);
            }
        }
    }

    fieldRawData.uploadToDevice(defaultStream);

    _setupArrayTexture(fieldRawData.devPtr());
}

} // namespace mirheo
