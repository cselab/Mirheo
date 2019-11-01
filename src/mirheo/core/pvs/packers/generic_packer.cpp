#include "generic_packer.h"

namespace mirheo
{

void GenericPacker::updateChannels(DataManager& manager, PackPredicate& predicate, cudaStream_t stream)
{
    nChannels = 0;
    bool needUpload = false;

    for (const auto& nameDesc : manager.getSortedChannels())
    {
        auto desc = nameDesc.second;
        
        if (!predicate(nameDesc)) continue;

        debug2("Packer: adding channel '%s' (id %d)\n",
               nameDesc.first.c_str(), nChannels);

        auto varPtr = getDevPtr(desc->varDataPtr);

        registerChannel(varPtr, desc->needShift(), needUpload, stream);
    }

    if (needUpload)
    {
        channelData  .uploadToDevice(stream);
        needShiftData.uploadToDevice(stream);
    }

    varChannelData = channelData  .devPtr();
    needShift      = needShiftData.devPtr();
}

GenericPackerHandler& GenericPacker::handler()
{
    return *static_cast<GenericPackerHandler*> (this);
}

void GenericPacker::registerChannel(CudaVarPtr varPtr, bool needShift,
                                    bool& needUpload, cudaStream_t stream)
{
    if (static_cast<int>(channelData.size()) <= nChannels)
    {
        channelData  .resize(nChannels+1, stream);
        needShiftData.resize(nChannels+1, stream);
        needUpload = true;
    }

    cuda_variant::apply_visitor([&](auto ptr)
    {
        using T = typename std::remove_pointer<decltype(ptr)>::type;

        if (cuda_variant::holds_alternative<T*> (channelData[nChannels]))
        {
            T *other = cuda_variant::get<T*> (channelData[nChannels]);
            if (other != ptr || needShiftData[nChannels] != needShift)
                needUpload = true;
        }
        else
        {
            needUpload = true;
        }

    }, varPtr);

    channelData  [nChannels] = varPtr;
    needShiftData[nChannels] = needShift;
    
    ++nChannels;
}

size_t GenericPacker::getSizeBytes(int numElements) const
{
    size_t size = 0;
    for (auto varPtr : channelData)
    {
        cuda_variant::apply_visitor([&](auto ptr)
        {
            using T = typename std::remove_pointer<decltype(ptr)>::type;
            size += getPaddedSize<T>(numElements);
        }, varPtr);
    }
    return size;
}

} // namespace mirheo
