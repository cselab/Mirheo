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
    inline __D__ size_t pack(int srcId, int dstId, char *dstBuffer, int numElements) const
    {
        TransformNone t;
        return pack(t, srcId, dstId, dstBuffer, numElements);
    }

    inline __D__ size_t packShift(int srcId, int dstId, char *dstBuffer, int numElements,
                                  float3 shift) const
    {
        TransformShift t {shift};
        return pack(t, srcId, dstId, dstBuffer, numElements);
    }


    inline __D__ size_t unpack(int srcId, int dstId, const char *srcBuffer, int numElements) const
    {
        TransformNone t;
        return unpack(t, srcId, dstId, srcBuffer, numElements);
    }

    inline __D__ size_t unpackShift(int srcId, int dstId, const char *srcBuffer, int numElements,
                                    float3 shift) const
    {
        TransformShift t {shift};
        return unpack(t, srcId, dstId, srcBuffer, numElements);
    }

    inline __D__ size_t unpackAtomicAddNonZero(int srcId, int dstId,
                                               const char *srcBuffer, int numElements,
                                               float eps) const
    {
        TransformAtomicAdd t {eps};
        return unpack(t, srcId, dstId, srcBuffer, numElements);
    }

    inline __D__ void copyTo(GenericPackerHandler& dst, int srcId, int dstId)
    {
        assert (nChannels == dst.nChannels);
        
        for (int i = 0; i < nChannels; ++i)
        {
            cuda_variant::apply_visitor([&](auto srcPtr)
            {
                using T = typename std::remove_pointer<decltype(srcPtr)>::type;
                auto dstPtr = cuda_variant::get_alternative<T*>(dst.varChannelData[i]);

                dstPtr[dstId]= srcPtr[srcId];
                
            }, varChannelData[i]);
        }
    }

    inline __D__ size_t getSizeBytes(int numElements) const
    {
        size_t sz = 0;

        for (int i = 0; i < nChannels; ++i)
        {
            cuda_variant::apply_visitor([&](auto srcPtr)
            {
                using T = typename std::remove_pointer<decltype(srcPtr)>::type;
                sz += getPaddedSize<T>(numElements);
            }, varChannelData[i]);
        }
        return sz;
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

    template <class Transform, typename T>
    inline __D__ size_t packElement(const Transform& transform, const T& val, int dstId,
                                    char *dstBuffer, int numElements) const
    {
        auto buffStart = reinterpret_cast<T*>(dstBuffer);
        transform( &buffStart[dstId], val );
        return getPaddedSize<T>(numElements);
    }

    template <class Transform>
    inline __D__ size_t pack(const Transform& transform, int srcId, int dstId,
                             char *dstBuffer, int numElements) const
    {
        size_t totPacked = 0;
        for (int i = 0; i < nChannels; ++i)
        {
            cuda_variant::apply_visitor([&](auto srcPtr)
            {
                using T = typename std::remove_pointer<decltype(srcPtr)>::type;
                totPacked += packElement(transform, srcPtr[srcId],
                                         dstId, dstBuffer + totPacked, numElements);
            }, varChannelData[i]);
        }

        return totPacked;
    }

    template <class Transform, typename T>
    inline __D__ size_t unpackElement(const Transform& transform, int srcId, T& val,
                                      const char *srcBuffer, int numElements) const
    {
        auto buffStart = reinterpret_cast<const T*>(srcBuffer);
        transform( &val, buffStart[srcId] );
        return getPaddedSize<T>(numElements);
    }
    
    template <class Transform>
    inline __D__ size_t unpack(const Transform& transform, int srcId, int dstId,
                               const char *srcBuffer, int numElements) const
    {
        size_t totPacked = 0;
        for (int i = 0; i < nChannels; i++)
        {
            cuda_variant::apply_visitor([&](auto dstPtr)
            {
                using T = typename std::remove_pointer<decltype(dstPtr)>::type;
                totPacked += unpackElement(transform, srcId, dstPtr[dstId],
                                           srcBuffer + totPacked, numElements);
            }, varChannelData[i]);
        }

        return totPacked;
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

    size_t getSizeBytes(int numElements) const;
    
protected:

    void registerChannel(CudaVarPtr varPtr, bool& needUpload, cudaStream_t stream);
    
    PinnedBuffer<CudaVarPtr> channelData;
};
