// Copyright 2020 ETH Zurich. All Rights Reserved.
#include "interface.h"

#include <texture_types.h>
#include <mirheo/core/utils/cuda_common.h>

namespace mirheo
{

ScalarField::ScalarField(const MirState *state, std::string name, real3 hField, real3 margin) :
    MirSimulationObject(state, name),
    fieldArray_(nullptr),
    margin3_(margin)
{
    // We'll make sdf a bit bigger, so that particles that flew away
    // would also be correctly bounced back
    extendedDomainSize_ = state->domain.localSize + 2.0_r * margin3_;
    resolution_         = make_int3( math::ceil(extendedDomainSize_ / hField) );
    h_                  = extendedDomainSize_ / make_real3(resolution_-1);
    invh_               = 1.0_r / h_;
}

ScalarField::~ScalarField()
{
    if (fieldArray_) {
        CUDA_Check( cudaFreeArray(fieldArray_) );
        CUDA_Check( cudaDestroyTextureObject(fieldTex_) );
    }
}

ScalarField::ScalarField(ScalarField&&) = default;

const ScalarFieldDeviceHandler& ScalarField::handler() const
{
    return *(ScalarFieldDeviceHandler*)this;
}

void ScalarField::_setupArrayTexture(const float *fieldDevPtr)
{
    debug("setting up cuda array and texture object for field '%s'", getCName());

    // Prepare array to be transformed into texture
    auto chDesc = cudaCreateChannelDesc<float>();
    CUDA_Check( cudaMalloc3DArray(&fieldArray_, &chDesc, make_cudaExtent(resolution_.x, resolution_.y, resolution_.z)) );

    cudaMemcpy3DParms copyParams = {};
    copyParams.srcPtr   = make_cudaPitchedPtr((void*)fieldDevPtr, resolution_.x*sizeof(float), resolution_.x, resolution_.y);
    copyParams.dstArray = fieldArray_;
    copyParams.extent   = make_cudaExtent(resolution_.x, resolution_.y, resolution_.z);
    copyParams.kind     = cudaMemcpyDeviceToDevice;

    CUDA_Check( cudaMemcpy3D(&copyParams) );

    // Create texture
    cudaResourceDesc resDesc = {};
    resDesc.resType         = cudaResourceTypeArray;
    resDesc.res.array.array = fieldArray_;

    cudaTextureDesc texDesc = {};
    texDesc.addressMode[0]   = cudaAddressModeWrap;
    texDesc.addressMode[1]   = cudaAddressModeWrap;
    texDesc.addressMode[2]   = cudaAddressModeWrap;
    texDesc.filterMode       = cudaFilterModePoint;
    texDesc.readMode         = cudaReadModeElementType;
    texDesc.normalizedCoords = 0;

    CUDA_Check( cudaCreateTextureObject(&fieldTex_, &resDesc, &texDesc, nullptr) );

    CUDA_Check( cudaDeviceSynchronize() );
}

} // namespace mirheo
