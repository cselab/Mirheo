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

/// This is used to filter channels that need to be packed
using PackPredicate = std::function< bool (const DataManager::NamedChannelDesc&) >;

/** \brief A device-friendly structure that is used to pack and unpack multiple channels into a single buffer.
    
    Additionally to being packed and unpacked, the data can be shifted.
    This facilitate the exchange and redistribute operations.

    The packed channels are structured in a single buffer containing:
    1. The first channel data
    2. padding 
    3. The second channel data
    4. padding
    5. ...

    Hence, the number of elements must be known in advance before packing.
    This is generally not a limitation, as memory must be allocated before packing.
 */
struct GenericPackerHandler
{
    /// Alignment sufficient for all types used in channels. Useful for
    /// external codes that operate with Mirheo's packing functions. 
    static constexpr size_t alignment = getPaddedSize<char>(1);

    __D__ size_t pack(int srcId, int dstId, char *dstBuffer, int numElements) const
    {
        TransformNone t;
        return _pack(t, srcId, dstId, dstBuffer, numElements);
    }

    __D__ size_t packShift(int srcId, int dstId, char *dstBuffer, int numElements,
                           real3 shift) const
    {
        TransformShift t {shift, needShift_};
        return _pack(t, srcId, dstId, dstBuffer, numElements);
    }

    __D__ size_t unpack(int srcId, int dstId, const char *srcBuffer, int numElements) const
    {
        TransformNone t;
        return _unpack(t, srcId, dstId, srcBuffer, numElements);
    }

    __D__ size_t unpackAtomicAddNonZero(int srcId, int dstId,
                                        const char *srcBuffer, int numElements,
                                        real eps) const
    {
        TransformAtomicAdd t {eps};
        return _unpack(t, srcId, dstId, srcBuffer, numElements);
    }

    __D__ size_t unpackShift(int srcId, int dstId, const char *srcBuffer, int numElements, real3 shift) const
    {
        TransformShift t {shift, needShift_};
        return _unpack(t, srcId, dstId, srcBuffer, numElements);
    }

    __D__ void copyTo(GenericPackerHandler& dst, int srcId, int dstId) const
    {
        assert (nChannels_ == dst.nChannels_);
        
        for (int i = 0; i < nChannels_; ++i)
        {
            cuda_variant::apply_visitor([&](auto srcPtr)
            {
                using T = typename std::remove_pointer<decltype(srcPtr)>::type;
                auto dstPtr = cuda_variant::get_alternative<T*>(dst.varChannelData_[i]);
                
                dstPtr[dstId]= srcPtr[srcId];
                
            }, varChannelData_[i]);
        }
    }

    __D__ size_t getSizeBytes(int numElements) const
    {
        size_t sz = 0;

        for (int i = 0; i < nChannels_; ++i)
        {
            cuda_variant::apply_visitor([&](auto srcPtr)
            {
                using T = typename std::remove_pointer<decltype(srcPtr)>::type;
                sz += getPaddedSize<T>(numElements);
            }, varChannelData_[i]);
        }
        return sz;
    }

private:
    struct TransformNone
    {
        template <typename T>
        __D__ void operator()(T *addr, const T& val, __UNUSED int channelId) const
        {
            *addr = val;
        }
    };

    struct TransformShift
    {
        template <typename T>
        __D__ void operator()(T *addr, T val, int channelId) const
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
        __D__ void operator()(T *addr, T val, __UNUSED int channelId) const
        {
            TypeAtomicAdd::apply(addr, val, eps);
        }

        real eps;
    };

    template <class Transform>
    __D__ size_t _pack(const Transform& transform, int srcId, int dstId,
                       char *dstBuffer, int numElements) const
    {
        size_t totPacked = 0;
        for (int i = 0; i < nChannels_; ++i)
        {
            cuda_variant::apply_visitor([&](auto srcPtr)
            {
                using T = typename std::remove_pointer<decltype(srcPtr)>::type;
                auto buffStart = reinterpret_cast<T*>(dstBuffer + totPacked);
                transform( &buffStart[dstId], srcPtr[srcId], i );
                totPacked += getPaddedSize<T>(numElements);
            }, varChannelData_[i]);
        }

        return totPacked;
    }

    template <class Transform>
    __D__ size_t _unpack(const Transform& transform, int srcId, int dstId,
                         const char *srcBuffer, int numElements) const
    {
        size_t totPacked = 0;
        for (int i = 0; i < nChannels_; i++)
        {
            cuda_variant::apply_visitor([&](auto dstPtr)
            {
                using T = typename std::remove_pointer<decltype(dstPtr)>::type;
                auto buffStart = reinterpret_cast<const T*>(srcBuffer + totPacked);
                transform( &dstPtr[dstId], buffStart[srcId], i );
                totPacked += getPaddedSize<T>(numElements);
            }, varChannelData_[i]);
        }

        return totPacked;
    }

protected:    
    int nChannels_              {0};       ///< number of data channels to pack / unpack
    CudaVarPtr *varChannelData_ {nullptr}; ///< device pointers of the packed data
    bool *needShift_            {nullptr}; ///< flag per channel: true if data needs to be shifted
};

class GenericPacker : private GenericPackerHandler
{
public:
    void updateChannels(DataManager& dataManager, PackPredicate& predicate, cudaStream_t stream);

    GenericPackerHandler& handler();

    size_t getSizeBytes(int numElements) const;
    
private:
    void _registerChannel(CudaVarPtr varPtr, bool needShift,
                          bool& needUpload, cudaStream_t stream);
    
    PinnedBuffer<CudaVarPtr> channelData_;
    PinnedBuffer<bool> needShiftData_;
};

} // namespace mirheo
