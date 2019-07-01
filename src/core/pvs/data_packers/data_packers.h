#pragma once

#include "utils.h"

#include <core/pvs/data_manager.h>
#include <core/utils/cpu_gpu_defines.h>
#include <core/utils/type_shift.h>

#include <cassert>
#include <functional>

using PackPredicate = std::function< bool (const DataManager::NamedChannelDesc&) >;

struct GenericData
{
    int nChannels              {0};        ///< number of data channels to pack / unpack
    CudaVarPtr *varChannelData {nullptr};  ///< device pointers of the packed data

    inline __D__ void pack(int srcId, int dstId, char *dstBuffer, int numElements) const
    {
        TransformNone t;
        pack(t, srcId, dstId, dstBuffer, numElements);
    }

    inline __D__ void packShift(int srcId, int dstId, char *dstBuffer, int numElements,
                                float3 shift) const
    {
        TransformShift t {shift};
        pack(t, srcId, dstId, dstBuffer, numElements);
    }

private:

    struct TransformNone
    {
        template <typename T>
        inline __D__ T operator()(const T& val) const {return val;}
    };

    struct TransformShift
    {
        template <typename T>
        inline __D__ T operator()(T val) const
        {
            TypeShift::apply(val, shift);
            return val;
        }

        float3 shift;
    };

    template <class Transform>
    inline __D__ void pack(const Transform& transform, int srcId, int dstId,
                           char *dstBuffer, int numElements) const
    {
        for (int i = 0; i < nChannels; ++i)
        {
            cuda_variant::apply_visitor([&](auto srcPtr)
            {
                using T = typename std::remove_pointer<decltype(srcPtr)>::type;
                auto buffStart = reinterpret_cast<T*>(dstBuffer);
                buffStart[dstId] = transform( srcPtr[srcId] );
                dstBuffer += getPaddedSize<T>(numElements);
            }, varChannelData[i]);
        }
    }

    template <class Transform>
    inline __D__ void unpack(const Transform& transform, int srcId, int dstId,
                             const char *srcBuffer, int numElements) const
    {
        for (int i = 0; i < nChannels; i++)
        {
            cuda_variant::apply_visitor([&](auto dstPtr)
            {
                using T = typename std::remove_pointer<decltype(dstPtr)>::type;
                auto buffStart = reinterpret_cast<const T*>(srcBuffer);
                dstPtr[dstId] = transform( buffStart[srcId] );
                srcBuffer += getPaddedSize<T>(numElements);
            }, varChannelData[i]);
        }
    }
};

class ParticlePacker
{
protected:
    GenericData particleData;
};

class ObjectPacker : public ParticlePacker
{
protected:
    GenericData objectData;
};

class RodPacker : public ObjectPacker
{
protected:
    GenericData bisegmentData;
};
