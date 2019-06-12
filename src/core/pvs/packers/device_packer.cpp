#include "device_packer.h"

void DevicePacker::registerChannel(DataManager& manager, CudaVarPtr varPtr, bool& needUpload, cudaStream_t stream)
{
    if (manager.channelPtrs.size() <= nChannels)
    {
        manager.channelPtrs.resize(nChannels+1, stream);
        needUpload = true;
    }

    cuda_variant::apply_visitor([&](auto ptr)
    {
        using T = typename std::remove_pointer<decltype(ptr)>::type;
        packedSize_byte += sizeof(T);

        if (cuda_variant::holds_alternative<T*> (manager.channelPtrs[nChannels]))
        {
            T *other = cuda_variant::get<T*> (manager.channelPtrs[nChannels]);
            if (other != ptr)
                needUpload = true;
        }
        else
            needUpload = true;

    }, varPtr);

    manager.channelPtrs[nChannels] = varPtr;
    
    ++nChannels;
}

void DevicePacker::registerChannels(PackPredicate predicate, DataManager& manager, const std::string& pvName, bool& needUpload, cudaStream_t stream)
{
    for (const auto& name_desc : manager.getSortedChannels())
    {
        auto desc = name_desc.second;
        
        if (!predicate(name_desc)) continue;

        auto varPtr = getDevPtr(desc->varDataPtr);

        registerChannel(manager, varPtr, needUpload, stream);
    }
}

void DevicePacker::setAndUploadData(DataManager& manager, bool needUpload, cudaStream_t stream)
{
    packedSize_byte = ( (packedSize_byte + sizeof(float4) - 1) / sizeof(float4) ) * sizeof(float4);

    if (needUpload)
        manager.channelPtrs.uploadToDevice(stream);

    channelData = manager.channelPtrs.devPtr();
}
