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

    /** \brief Fetch one datum from the registered channels and pack it into a buffer
        \param [in] srcId Index of the datum to fetch from registered channel space (in number of elements).
        \param [in] dstId Index of the datum to store in dstBuffer space (in number of elements).
        \param [out] dstBuffer Destination buffer
        \param [in] numElements Total number of elements that will be packed in the buffer.
        \return The size (in bytes) taken by the packed data (numElements elements)
     */
    __D__ size_t pack(int srcId, int dstId, char *dstBuffer, int numElements) const
    {
        TransformNone t;
        return _pack(t, srcId, dstId, dstBuffer, numElements);
    }

    /** \brief Fetch one datum from the registered channels, shift it (if applicable) and pack it into the buffer
        \param [in] srcId Index of the datum to fetch from registered channel space (in number of elements).
        \param [in] dstId Index of the datum to store in dstBuffer space (in number of elements).
        \param [out] dstBuffer Destination buffer
        \param [in] numElements Total number of elements that will be packed in the buffer.
        \param [in] shift The coordinate shift
        \return The size (in bytes) taken by the packed data (numElements elements)

        Only channels with active shift will be shifted.
     */
    __D__ size_t packShift(int srcId, int dstId, char *dstBuffer, int numElements,
                           real3 shift) const
    {
        TransformShift t {shift, needShift_};
        return _pack(t, srcId, dstId, dstBuffer, numElements);
    }

    /** \brief Unpack one datum from the buffer and store it in the registered channels
        \param [in] srcId Index of the datum to fetch from the buffer (in number of elements).
        \param [in] dstId Index of the datum to store in the registered channels (in number of elements).
        \param [in] srcBuffer Source buffer that contains packed data.
        \param [in] numElements Total number of elements that are packed in the buffer.
        \return The size (in bytes) taken by the packed data (numElements elements)
     */
    __D__ size_t unpack(int srcId, int dstId, const char *srcBuffer, int numElements) const
    {
        TransformNone t;
        return _unpack(t, srcId, dstId, srcBuffer, numElements);
    }

    /** \brief Unpack one datum from the buffer and add it to the registered channels atomically.
        \param [in] srcId Index of the datum to fetch from the buffer (in number of elements).
        \param [in] dstId Index of the datum to add to the registered channels (in number of elements).
        \param [in] srcBuffer Source buffer that contains packed data.
        \param [in] numElements Total number of elements that are packed in the buffer.
        \param [in] eps Only elements that are larger than this tolerance will be added.
        \return The size (in bytes) taken by the packed data (numElements elements)
     */
    __D__ size_t unpackAtomicAddNonZero(int srcId, int dstId,
                                        const char *srcBuffer, int numElements,
                                        real eps) const
    {
        TransformAtomicAdd t {eps};
        return _unpack(t, srcId, dstId, srcBuffer, numElements);
    }

    /** \brief Unpack and shift one datum from the buffer and store it in the registered channels.
        \param [in] srcId Index of the datum to fetch from the buffer (in number of elements).
        \param [in] dstId Index of the datum to store into the registered channels (in number of elements).
        \param [in] srcBuffer Source buffer that contains packed data.
        \param [in] numElements Total number of elements that are packed in the buffer.
        \param [in] shift The coordinate shift
        \return The size (in bytes) taken by the packed data (numElements elements)
     */
    __D__ size_t unpackShift(int srcId, int dstId, const char *srcBuffer, int numElements, real3 shift) const
    {
        TransformShift t {shift, needShift_};
        return _unpack(t, srcId, dstId, srcBuffer, numElements);
    }

    /** \brief Copy one datum from the registered channels to the registered channels of another GenericPackerHandler.
        \param [in] dst The other GenericPackerHandler that will receive the new datum.
        \param [in] srcId Index of the datum to fetch from the registered channels (in number of elements).
        \param [in] dstId Index of the datum to store into the dst registered channels (in number of elements).
     */
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

    /** \brief Get the size (in bytes) of the buffer that can hold the packed data of numElements elements from all registered channels.
        \param [in] numElements The number of elements that the buffer must contain once packed.
        \return The size (in bytes) of the buffer.

        This must be used to allocate the buffer size. 
        Because of padding, the size is not simply the sum of sizes of all elements.
     */
    __HD__ size_t getSizeBytes(int numElements) const
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

/// \brief This class is used to construct GenericPackerHandler, to be passed to the device.
class GenericPacker : private GenericPackerHandler
{
public:
    /** \brief Register all channels of a DataManager satisfying a predicate.
        \param [in] dataManager The object that contains the channels to register
        \param [in] predicate The filter (white list) that is used to select the channels to register, 
                              based on their description and names
        \param [in] stream The stream used to transfer the data on the device
        
        All previously registered channels will be removed before adding those described above.
     */
    void updateChannels(DataManager& dataManager, PackPredicate& predicate, cudaStream_t stream);

    /// Get a handler that can be used on the device.
    GenericPackerHandler& handler();

    /// see GenericPackerHandler::getSizeBytes().
    size_t getSizeBytes(int numElements) const;
    
private:
    void _registerChannel(CudaVarPtr varPtr, bool needShift,
                          bool& needUpload, cudaStream_t stream);
    
    PinnedBuffer<CudaVarPtr> channelData_;
    PinnedBuffer<bool> needShiftData_;
};

} // namespace mirheo
