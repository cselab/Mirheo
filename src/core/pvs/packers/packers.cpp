#include "packers.h"

ParticlePacker::ParticlePacker(ParticleVector *pv, LocalParticleVector *lpv, PackPredicate predicate, cudaStream_t stream)
{
    if (pv == nullptr || lpv == nullptr) return;

    auto& manager = lpv->dataPerParticle;

    bool needUpload = false;

    registerChannel(manager, sizeof(float4), reinterpret_cast<char*>(lpv->positions() .devPtr()), needUpload, stream);
    registerChannel(manager, sizeof(float4), reinterpret_cast<char*>(lpv->velocities().devPtr()), needUpload, stream);

    registerChannels(predicate, manager, pv->name, needUpload, stream);
    setAndUploadData(           manager,           needUpload, stream);
}

ParticleExtraPacker::ParticleExtraPacker(ParticleVector *pv, LocalParticleVector *lpv, PackPredicate predicate, cudaStream_t stream)
{
    if (pv == nullptr || lpv == nullptr) return;

    auto& manager = lpv->dataPerParticle;

    bool needUpload = false;
    registerChannels(predicate, manager, pv->name, needUpload, stream);
    setAndUploadData(           manager,           needUpload, stream);
}


ObjectExtraPacker::ObjectExtraPacker(ObjectVector *ov, LocalObjectVector *lov, PackPredicate predicate, cudaStream_t stream)
{
    if (ov == nullptr || lov == nullptr) return;

    auto& manager = lov->dataPerObject;

    bool needUpload = false;
    registerChannels(predicate, manager, ov->name, needUpload, stream);
    setAndUploadData(           manager,           needUpload, stream);
}


ObjectPacker::ObjectPacker(ObjectVector *ov, LocalObjectVector *lov, PackPredicate predicate, cudaStream_t stream) :
    part(ov, lov, predicate, stream),
    obj (ov, lov, predicate, stream)
{
    if (ov == nullptr || lov == nullptr) return;
    totalPackedSize_byte = part.packedSize_byte * ov->objSize + obj.packedSize_byte;
}
