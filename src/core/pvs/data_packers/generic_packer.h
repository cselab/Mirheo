#pragma once

#include "utils.h"

#include <core/pvs/data_manager.h>
#include <core/utils/cpu_gpu_defines.h>
#include <core/utils/type_add.h>
#include <core/utils/type_shift.h>

#include <cassert>
#include <functional>

using PackPredicate = std::function< bool (const DataManager::NamedChannelDesc&) >;

struct GenericPackerHandler
{
    inline __D__ char* pack(int srcId, int dstId, char *dstBuffer, int numElements) const
    {
        TransformNone t;
        return pack(t, srcId, dstId, dstBuffer, numElements);
    }

    inline __D__ char* packShift(int srcId, int dstId, char *dstBuffer, int numElements,
                                float3 shift) const
    {
        TransformShift t {shift};
        return pack(t, srcId, dstId, dstBuffer, numElements);
    }


    inline __D__ const char* unpack(int srcId, int dstId, const char *srcBuffer, int numElements) const
    {
        TransformNone t;
        return unpack(t, srcId, dstId, srcBuffer, numElements);
    }

    inline __D__ const char* unpackShift(int srcId, int dstId, const char *srcBuffer, int numElements,
                                  float3 shift) const
    {
        TransformShift t {shift};
        return unpack(t, srcId, dstId, srcBuffer, numElements);
    }

    inline __D__ const char* unpackAtomicAddNonZero(int srcId, int dstId,
                                                    const char *srcBuffer, int numElements,
                                                    float eps) const
    {
        TransformAtomicAdd t {eps};
        return unpack(t, srcId, dstId, srcBuffer, numElements);
    }

private:

    struct TransformNone
    {
        template <typename T>
        inline __D__ void operator()(T *addr, const T& val) const {*addr = val;}
    };

    struct TransformShift
    {
        template <typename T>
        inline __D__ void operator()(T *addr, T val) const
        {
            TypeShift::apply(val, shift);
            *addr = val;
        }

        float3 shift;
    };

    struct TransformAtomicAdd
    {
        template <typename T>
        inline __D__ void operator()(T *addr, T val) const
        {
            TypeAtomicAdd::apply(addr, val, eps);
            *addr = val;
        }

        float eps;
    };

    template <class Transform>
    inline __D__ char* pack(const Transform& transform, int srcId, int dstId,
                           char *dstBuffer, int numElements) const
    {
        for (int i = 0; i < nChannels; ++i)
        {
            cuda_variant::apply_visitor([&](auto srcPtr)
            {
                using T = typename std::remove_pointer<decltype(srcPtr)>::type;
                auto buffStart = reinterpret_cast<T*>(dstBuffer);
                transform( &buffStart[dstId], srcPtr[srcId] );
                dstBuffer += getPaddedSize<T>(numElements);
            }, varChannelData[i]);
        }

        return dstBuffer;
    }

    template <class Transform>
    inline __D__ const char* unpack(const Transform& transform, int srcId, int dstId,
                                    const char *srcBuffer, int numElements) const
    {
        for (int i = 0; i < nChannels; i++)
        {
            cuda_variant::apply_visitor([&](auto dstPtr)
            {
                using T = typename std::remove_pointer<decltype(dstPtr)>::type;
                auto buffStart = reinterpret_cast<const T*>(srcBuffer);
                transform( &dstPtr[dstId], buffStart[srcId] );
                srcBuffer += getPaddedSize<T>(numElements);
            }, varChannelData[i]);
        }

        return srcBuffer;
    }


    
protected:

    int nChannels              {0};        ///< number of data channels to pack / unpack
    CudaVarPtr *varChannelData {nullptr};  ///< device pointers of the packed data
};

class GenericPacker : public GenericPackerHandler
{
public:
    void updateChannels(DataManager& dataManager, PackPredicate& predicate, cudaStream_t stream);

    GenericPackerHandler& handler();
    
protected:

    void registerChannel(CudaVarPtr varPtr, bool& needUpload, cudaStream_t stream);
    
    PinnedBuffer<CudaVarPtr> channelData;
};
