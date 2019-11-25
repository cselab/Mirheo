#pragma once

#include "utils.h"

#include <mirheo/core/pvs/data_manager.h>
#include <mirheo/core/utils/cpu_gpu_defines.h>
#include <mirheo/core/utils/type_add.h>
#include <mirheo/core/utils/type_shift.h>

#include <cassert>
#include <functional>

namespace mirheo
{

using PackPredicate = std::function< bool (const DataManager::NamedChannelDesc&) >;

struct GenericPackerHandler
{
    /// Alignment sufficient for all types used in channels. Useful for
    /// external codes that operate with Mirheo's packing functions. 
    static constexpr size_t alignment = getPaddedSize<char>(1);

    inline __D__ size_t pack(int srcId, int dstId, char *dstBuffer, int numElements) const
    {
        TransformNone t;
        return pack(t, srcId, dstId, dstBuffer, numElements);
    }

    inline __D__ size_t packShift(int srcId, int dstId, char *dstBuffer, int numElements,
                                  real3 shift) const
    {
        TransformShift t {shift, needShift};
        return pack(t, srcId, dstId, dstBuffer, numElements);
    }

    inline __D__ size_t unpack(int srcId, int dstId, const char *srcBuffer, int numElements) const
    {
        TransformNone t;
        return unpack(t, srcId, dstId, srcBuffer, numElements);
    }

    inline __D__ size_t unpackAtomicAddNonZero(int srcId, int dstId,
                                               const char *srcBuffer, int numElements,
                                               real eps) const
    {
        TransformAtomicAdd t {eps};
        return unpack(t, srcId, dstId, srcBuffer, numElements);
    }

    inline __D__ size_t unpackShift(int srcId, int dstId, const char *srcBuffer, int numElements, real3 shift) const
    {
        TransformShift t {shift, needShift};
        return unpack(t, srcId, dstId, srcBuffer, numElements);
    }

    inline __D__ void copyTo(GenericPackerHandler& dst, int srcId, int dstId) const
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
        inline __D__ void operator()(T *addr, const T& val, __UNUSED int channelId) const
        {
            *addr = val;
        }
    };

    struct TransformShift
    {
        template <typename T>
        inline __D__ void operator()(T *addr, T val, int channelId) const
        {
            if (needShift[channelId])
                TypeShift::apply(val, shift);
            *addr = val;
        }

        real3 shift;
        const bool *needShift;
    };

    struct TransformAtomicAdd
    {
        template <typename T>
        inline __D__ void operator()(T *addr, T val, __UNUSED int channelId) const
        {
            TypeAtomicAdd::apply(addr, val, eps);
        }

        real eps;
    };

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
                auto buffStart = reinterpret_cast<T*>(dstBuffer + totPacked);
                transform( &buffStart[dstId], srcPtr[srcId], i );
                totPacked += getPaddedSize<T>(numElements);
            }, varChannelData[i]);
        }

        return totPacked;
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
                auto buffStart = reinterpret_cast<const T*>(srcBuffer + totPacked);
                transform( &dstPtr[dstId], buffStart[srcId], i );
                totPacked += getPaddedSize<T>(numElements);
            }, varChannelData[i]);
        }

        return totPacked;
    }
    
protected:

    int nChannels              {0};       ///< number of data channels to pack / unpack
    CudaVarPtr *varChannelData {nullptr}; ///< device pointers of the packed data
    bool *needShift            {nullptr}; ///< flag per channel: true if data needs to be shifted
};

class GenericPacker : public GenericPackerHandler
{
public:
    void updateChannels(DataManager& dataManager, PackPredicate& predicate, cudaStream_t stream);

    GenericPackerHandler& handler();

    size_t getSizeBytes(int numElements) const;
    
protected:

    void registerChannel(CudaVarPtr varPtr,  bool needShift,
                         bool& needUpload, cudaStream_t stream);
    
    PinnedBuffer<CudaVarPtr> channelData;
    PinnedBuffer<bool> needShiftData;
};

} // namespace mirheo
