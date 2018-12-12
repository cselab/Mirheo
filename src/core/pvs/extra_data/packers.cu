#include "packers.h"

ParticlePacker::ParticlePacker(ParticleVector* pv, LocalParticleVector* lpv, cudaStream_t stream)
{
    if (pv == nullptr || lpv == nullptr) return;

    auto& manager = lpv->extraPerParticle;

    int n = 0;
    bool upload = false;

    auto registerChannel = [&] (int sz, char* ptr, int typesize) {

        if (manager.channelPtrs.size() <= n)
        {
            manager.channelPtrs.        resize(n+1, stream);
            manager.channelSizes.       resize(n+1, stream);
            manager.channelShiftTypes.  resize(n+1, stream);

            upload = true;
        }

        if (ptr != manager.channelPtrs[n]) upload = true;

        manager.channelSizes[n] = sz;
        manager.channelPtrs[n] = ptr;
        manager.channelShiftTypes[n] = typesize;

        packedSize_byte += sz;
        n++;
    };

    registerChannel(
                    sizeof(Particle),
                    reinterpret_cast<char*>(lpv->coosvels.devPtr()),
                    sizeof(float) );


    for (const auto& name_desc : manager.getSortedChannels())
    {
        auto desc = name_desc.second;

        if (desc->communication == ExtraDataManager::CommunicationMode::NeedExchange)
        {
            int sz = desc->container->datatype_size();

            if (sz % sizeof(int) != 0)
                die("Size of extra data per particle should be divisible by 4 bytes (PV '%s', data entry '%s')",
                    pv->name.c_str(), name_desc.first.c_str());

            if ( sz % sizeof(float4) && (desc->shiftTypeSize == 4 || desc->shiftTypeSize == 8) )
                die("Size of extra data per particle should be divisible by 16 bytes"
                    "when shifting is required (PV '%s', data entry '%s')",
                    pv->name.c_str(), name_desc.first.c_str());

            registerChannel(
                            sz,
                            reinterpret_cast<char*>(desc->container->genericDevPtr()),
                            desc->shiftTypeSize);
        }
    }

    nChannels = n;
    packedSize_byte = ( (packedSize_byte + sizeof(float4) - 1) / sizeof(float4) ) * sizeof(float4);

    if (upload)
    {
        manager.channelPtrs.        uploadToDevice(stream);
        manager.channelSizes.       uploadToDevice(stream);
        manager.channelShiftTypes.  uploadToDevice(stream);
    }

    channelData         = manager.channelPtrs.        devPtr();
    channelSizes        = manager.channelSizes.       devPtr();
    channelShiftTypes   = manager.channelShiftTypes.  devPtr();
}

ObjectExtraPacker::ObjectExtraPacker(ObjectVector* ov, LocalObjectVector* lov, cudaStream_t stream)
{
    if (ov == nullptr || lov == nullptr) return;

    auto& manager = lov->extraPerObject;

    int n = 0;
    bool upload = false;

    auto registerChannel = [&] (int sz, char* ptr, int typesize) {

        if (manager.channelPtrs.size() <= n)
        {
            manager.channelPtrs.        resize(n+1, stream);
            manager.channelSizes.       resize(n+1, stream);
            manager.channelShiftTypes.  resize(n+1, stream);

            upload = true;
        }

        if (ptr != manager.channelPtrs[n]) upload = true;

        manager.channelSizes[n] = sz;
        manager.channelPtrs[n] = ptr;
        manager.channelShiftTypes[n] = typesize;

        packedSize_byte += sz;
        n++;
    };

    for (const auto& name_desc : manager.getSortedChannels())
    {
        auto desc = name_desc.second;

        if (desc->communication == ExtraDataManager::CommunicationMode::NeedExchange)
        {
            int sz = desc->container->datatype_size();

            if (sz % sizeof(int) != 0)
                die("Size of extra data per particle should be divisible by 4 bytes (PV '%s', data entry '%s')",
                    ov->name.c_str(), name_desc.first.c_str());

            if ( sz % sizeof(float4) && (desc->shiftTypeSize == 4 || desc->shiftTypeSize == 8) )
                die("Size of extra data per particle should be divisible by 16 bytes"
                    "when shifting is required (PV '%s', data entry '%s')",
                    ov->name.c_str(), name_desc.first.c_str());

            registerChannel(
                            sz,
                            reinterpret_cast<char*>(desc->container->genericDevPtr()),
                            desc->shiftTypeSize);
        }
    }

    nChannels = n;
    packedSize_byte = ( (packedSize_byte + sizeof(float4) - 1) / sizeof(float4) ) * sizeof(float4);

    if (upload)
    {
        manager.channelPtrs.        uploadToDevice(stream);
        manager.channelSizes.       uploadToDevice(stream);
        manager.channelShiftTypes.  uploadToDevice(stream);
    }

    channelData         = manager.channelPtrs.        devPtr();
    channelSizes        = manager.channelSizes.       devPtr();
    channelShiftTypes   = manager.channelShiftTypes.  devPtr();
}


ObjectPacker::ObjectPacker(ObjectVector* ov, LocalObjectVector* lov, cudaStream_t stream) :
    part(ov, lov, stream), obj(ov, lov, stream)
{
    if (ov == nullptr || lov == nullptr) return;
    totalPackedSize_byte = part.packedSize_byte * ov->objSize + obj.packedSize_byte;
}
