#pragma once

#include "map.h"

#include <core/pvs/data_manager.h>
#include <core/utils/cpu_gpu_defines.h>

#include <vector_types.h>

template <typename T>
class _DevicePacker
{
    _DevicePacker(const MapEntry *map, DataManager::ChannelDescription& channelDesc) noexcept :
        map(map)
    {
        using BuffType = PinnedBuffer<T>*;

        if (!mpark::holds_alternative< BuffType >(channelDesc.varDataPtr))
            die("Channel is holding a different type than the required one");

        channelData = mpark::get<BuffType>(channelDesc.varDataPtr)->devPtr();
    }
        
    __D__ inline void pack(int i, size_t offset, char *dstAddr) const
    {
        auto dst = (T*) dstAddr;
        int srcId = map[i];
        dst[i] = channelData[srcId];
    }

    __D__ inline void unpack(int i, const char *srcAddr)
    {
        auto src = (const T*) srcAddr;
        int dstId = map[i];
        channelData[dstId] = src[i];
    }

protected:
    const MapEntry *map;
    T *channelData;
};
