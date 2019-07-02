#include "generic_packer.h"

void GenericPacker::updateChannels(DataManager& manager, PackPredicate& predicate, cudaStream_t stream)
{
    nChannels = 0;
    bool needUpload = false;

    for (const auto& nameDesc : manager.getSortedChannels())
    {
        auto desc = nameDesc.second;
        
        if (!predicate(nameDesc)) continue;

        auto varPtr = getDevPtr(desc->varDataPtr);

        registerChannel(varPtr, needUpload, stream);
    }

    if (needUpload)
        channelData.uploadToDevice(stream);

    varChannelData = channelData.devPtr();
}

GenericPackerHandler& GenericPacker::handler()
{
    nChannels      = channelData.size();
    varChannelData = channelData.devPtr();
    
    return *static_cast<GenericPackerHandler*> (this);
}

void GenericPacker::registerChannel(CudaVarPtr varPtr, bool& needUpload, cudaStream_t stream)
{
    if (channelData.size() <= nChannels)
    {
        channelData.resize(nChannels+1, stream);
        needUpload = true;
    }

    cuda_variant::apply_visitor([&](auto ptr)
    {
        using T = typename std::remove_pointer<decltype(ptr)>::type;

        if (cuda_variant::holds_alternative<T*> (channelData[nChannels]))
        {
            T *other = cuda_variant::get<T*> (channelData[nChannels]);
            if (other != ptr)
                needUpload = true;
        }
        else
            needUpload = true;

    }, varPtr);

    channelData[nChannels] = varPtr;
    
    ++nChannels;
}
