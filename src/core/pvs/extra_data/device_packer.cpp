#include "device_packer.h"

void DevicePacker::registerChannel(ExtraDataManager& manager, int sz, char *ptr, int typesize, bool& needUpload, cudaStream_t stream)
{
    if (manager.channelPtrs.size() <= nChannels)
    {
        manager.channelPtrs.        resize(nChannels+1, stream);
        manager.channelSizes.       resize(nChannels+1, stream);
        manager.channelShiftTypes.  resize(nChannels+1, stream);

        needUpload = true;
    }

    if (ptr != manager.channelPtrs[nChannels])
        needUpload = true;

    manager.channelSizes[nChannels] = sz;
    manager.channelPtrs[nChannels] = ptr;
    manager.channelShiftTypes[nChannels] = typesize;

    packedSize_byte += sz;
    ++nChannels;
}

void DevicePacker::registerChannels(PackPredicate predicate, ExtraDataManager& manager, const std::string& pvName, bool& needUpload, cudaStream_t stream)
{
    for (const auto& name_desc : manager.getSortedChannels())
    {
        auto desc = name_desc.second;
        
        if (!predicate(name_desc)) continue;

        int sz = desc->container->datatype_size();

        if (sz % sizeof(int) != 0)
            die("Size of extra data per particle should be divisible by 4 bytes (PV '%s', data entry '%s')",
                pvName.c_str(), name_desc.first.c_str());

        if ( sz % sizeof(float4) && (desc->shiftTypeSize == 4 || desc->shiftTypeSize == 8) )
            die("Size of extra data per particle should be divisible by 16 bytes"
                "when shifting is required (PV '%s', data entry '%s')",
                pvName.c_str(), name_desc.first.c_str());

        registerChannel(manager, sz,
                        reinterpret_cast<char*>(desc->container->genericDevPtr()),
                        desc->shiftTypeSize, needUpload, stream);
    }
}

void DevicePacker::setAndUploadData(ExtraDataManager& manager, bool needUpload, cudaStream_t stream)
{
    packedSize_byte = ( (packedSize_byte + sizeof(float4) - 1) / sizeof(float4) ) * sizeof(float4);

    if (needUpload)
    {
        manager.channelPtrs.        uploadToDevice(stream);
        manager.channelSizes.       uploadToDevice(stream);
        manager.channelShiftTypes.  uploadToDevice(stream);
    }

    channelData         = manager.channelPtrs.        devPtr();
    channelSizes        = manager.channelSizes.       devPtr();
    channelShiftTypes   = manager.channelShiftTypes.  devPtr();
}
