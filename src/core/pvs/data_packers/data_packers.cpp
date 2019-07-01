#include "data_packers.h"


GenericData::GenericData()
{}

void GenericData::updateChannels(DataManager& dataManager, PackPredicate& predicate, cudaStream_t stream)
{
    
}

GenericDataHandler& GenericData::handler()
{
    nChannels      = channelData.size();
    varChannelData = channelData.devPtr();
    
    return *static_cast<GenericDataHandler*> (this);
}

void GenericData::registerChannel(DataManager& dataManager, CudaVarPtr varPtr, bool& needUpload, cudaStream_t stream)
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
