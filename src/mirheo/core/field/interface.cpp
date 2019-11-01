#include "interface.h"

#include <texture_types.h>
#include <mirheo/core/utils/cuda_common.h>

namespace mirheo
{

Field::Field(const MirState *state, std::string name, real3 hField) :
    MirSimulationObject(state, name),
    fieldArray(nullptr)
{
    // We'll make sdf a bit bigger, so that particles that flew away
    // would also be correctly bounced back
    extendedDomainSize = state->domain.localSize + 2.0_r * margin3;
    resolution         = make_int3( math::ceil(extendedDomainSize / hField) );
    h                  = extendedDomainSize / make_real3(resolution-1);
    invh               = 1.0_r / h;
}

Field::~Field()
{
    if (fieldArray) {
        CUDA_Check( cudaFreeArray(fieldArray) );
        CUDA_Check( cudaDestroyTextureObject(fieldTex) );
    }
}

Field::Field(Field&&) = default;

const FieldDeviceHandler& Field::handler() const
{
    return *(FieldDeviceHandler*)this;
}

void Field::setupArrayTexture(const float *fieldDevPtr)
{
    debug("setting up cuda array and texture object for field '%s'", name.c_str());
    
    // Prepare array to be transformed into texture
    auto chDesc = cudaCreateChannelDesc<float>();
    CUDA_Check( cudaMalloc3DArray(&fieldArray, &chDesc, make_cudaExtent(resolution.x, resolution.y, resolution.z)) );

    cudaMemcpy3DParms copyParams = {};
    copyParams.srcPtr   = make_cudaPitchedPtr((void*)fieldDevPtr, resolution.x*sizeof(float), resolution.x, resolution.y);
    copyParams.dstArray = fieldArray;
    copyParams.extent   = make_cudaExtent(resolution.x, resolution.y, resolution.z);
    copyParams.kind     = cudaMemcpyDeviceToDevice;

    CUDA_Check( cudaMemcpy3D(&copyParams) );

    // Create texture
    cudaResourceDesc resDesc = {};
    resDesc.resType         = cudaResourceTypeArray;
    resDesc.res.array.array = fieldArray;

    cudaTextureDesc texDesc = {};
    texDesc.addressMode[0]   = cudaAddressModeWrap;
    texDesc.addressMode[1]   = cudaAddressModeWrap;
    texDesc.addressMode[2]   = cudaAddressModeWrap;
    texDesc.filterMode       = cudaFilterModePoint;
    texDesc.readMode         = cudaReadModeElementType;
    texDesc.normalizedCoords = 0;

    CUDA_Check( cudaCreateTextureObject(&fieldTex, &resDesc, &texDesc, nullptr) );

    CUDA_Check( cudaDeviceSynchronize() );
}

} // namespace mirheo
