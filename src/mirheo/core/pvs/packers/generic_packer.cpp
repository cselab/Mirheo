// Copyright 2020 ETH Zurich. All Rights Reserved.
#include "generic_packer.h"

namespace mirheo
{

void GenericPacker::updateChannels(DataManager& manager, PackPredicate& predicate, cudaStream_t stream)
{
    nChannels_ = 0;
    bool needUpload = false;

    for (const auto& nameDesc : manager.getSortedChannels())
    {
        auto desc = nameDesc.second;

        if (!predicate(nameDesc)) continue;

        debug2("Packer: adding channel '%s' (id %d)",
               nameDesc.first.c_str(), nChannels_);

        auto varPtr = getDevPtr(desc->varDataPtr);

        _registerChannel(varPtr, desc->needShift(), needUpload, stream);
    }

    if (needUpload)
    {
        channelData_  .uploadToDevice(stream);
        needShiftData_.uploadToDevice(stream);
    }

    varChannelData_ = channelData_  .devPtr();
    needShift_      = needShiftData_.devPtr();
}

GenericPackerHandler& GenericPacker::handler()
{
    return *static_cast<GenericPackerHandler*> (this);
}

void GenericPacker::_registerChannel(CudaVarPtr varPtr, bool needShift,
                                     bool& needUpload, cudaStream_t stream)
{
    if (static_cast<int>(channelData_.size()) <= nChannels_)
    {
        channelData_  .resize(nChannels_+1, stream);
        needShiftData_.resize(nChannels_+1, stream);
        needUpload = true;
    }

    cuda_variant::apply_visitor([&](auto ptr)
    {
        using T = typename std::remove_pointer<decltype(ptr)>::type;

        if (cuda_variant::holds_alternative<T*> (channelData_[nChannels_]))
        {
            T *other = cuda_variant::get<T*> (channelData_[nChannels_]);
            if (other != ptr || needShiftData_[nChannels_] != needShift)
                needUpload = true;
        }
        else
        {
            needUpload = true;
        }

    }, varPtr);

    channelData_  [nChannels_] = varPtr;
    needShiftData_[nChannels_] = needShift;

    ++nChannels_;
}

size_t GenericPacker::getSizeBytes(int numElements) const
{
    size_t size = 0;
    for (auto varPtr : channelData_)
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
