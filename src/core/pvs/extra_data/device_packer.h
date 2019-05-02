#pragma once

#include "extra_data_manager.h"

#include <cassert>
#include <functional>

using PackPredicate = std::function< bool (const DataManager::NamedChannelDesc&) >;

/**
 * Class that packs nChannels of arbitrary data into a chunk of contiguous memory
 * or unpacks it in the same manner
 */
struct DevicePacker
{
    int packedSize_byte = 0;

    int nChannels = 0;                    ///< number of data channels to pack / unpack
    int *channelSizes        = nullptr;   ///< size if bytes of each channel entry, e.g. sizeof(Particle)
    int *channelShiftTypes   = nullptr;   ///< if type is 4, then treat data to shift as float3, if it is 8 -- as double3
    char **channelData       = nullptr;   ///< device pointers of the packed data

#ifdef __CUDACC__
    /**
     * Pack entity with id srcId into memory starting with dstAddr
     * with no shift
     */
    inline __device__ void pack(int srcId, char *dstAddr) const
    {
        _packShift<ShiftMode::NoShift> (srcId, dstAddr, make_float3(0.f, 0.f, 0.f));
    }

    /**
     * Pack entity with id srcId into memory starting with dstAddr
     * Apply shifts where needed
     */
    inline __device__ void packShift(int srcId, char *dstAddr, float3 shift) const
    {
        _packShift<ShiftMode::NeedShift>  (srcId, dstAddr, shift);
    }

    /**
     * Pack entity with id srcId into another packer at id dstId
     * with no shift
     * Assumes that the 2 packers contain the same channels
     */
    inline __device__ void pack(int srcId, DevicePacker& dst, int dstId) const
    {
        _packShift<ShiftMode::NoShift> (srcId, dst, dstId, make_float3(0.f, 0.f, 0.f));
    }

    /**
     * Pack entity with id srcId into another packer at id dstId
     * Apply shifts where needed
     * Assumes that the 2 packers contain the same channels
     */
    inline __device__ void packShift(int srcId, DevicePacker& dst, int dstId, float3 shift) const
    {
        _packShift<ShiftMode::NeedShift>  (srcId, dst, dstId, shift);
    }

    
    
    /**
     * Unpack entity from memory by srcAddr to the channels to id dstId
     */
    inline __device__ void unpack(const char *srcAddr, int dstId) const
    {
        for (int i = 0; i < nChannels; i++)
        {
            const int size = channelSizes[i];
            char *dstAddr = channelData[i] + size * dstId;
            copy(dstAddr, srcAddr, size);
            srcAddr += size;
        }
    }

    /**
     * atomicly add entity from memory by srcAddr to the channels to id dstId; assumes floating points
     */
    inline __device__ void unpackAtomicAdd(const char *srcAddr, int dstId) const
    {
        constexpr float tolerance = 1e-7;
        
        for (int i = 0; i < nChannels; i++)
        {
            const int size = channelSizes[i];
            char *dstAddr = channelData[i] + size * dstId;                        
            const float *src = (const float*) srcAddr;
            float       *dst = (      float*) dstAddr;
                
            for (int j = 0; j < size / sizeof(float); ++j) {
                float val = src[j];
                if (fabs(val) > tolerance)
                    atomicAdd(&dst[j], val);
            }
            srcAddr += size;
        }
    }

#endif /* __CUDACC__ */
    
private:

    enum class ShiftMode
    {
        NeedShift, NoShift
    };

#ifdef __CUDACC__
    /**
     * Copy nchunks*sizeof(T) bytes from \c from to \c to
     */
    template<typename T>
    inline __device__ void _copy(char *to, const char *from, int nchunks) const
    {
        auto _to   = (T*)to;
        auto _from = (const T*)from;

#pragma unroll 2
        for (int i = 0; i < nchunks; i++)
            _to[i] = _from[i];
    }

    /**
     * Copy size_bytes bytes from \c from to \c to
     * Speed up copying by choosing the widest possible data type
     * and calling the appropriate _copy function
     */
    inline __device__ void copy(char *to, const char *from, int size_bytes) const
    {
        assert(size_bytes % sizeof(int) == 0);

        if (size_bytes % sizeof(int4) == 0)
            _copy<int4>(to, from, size_bytes / sizeof(int4));
        else if (size_bytes % sizeof(int2) == 0)
            _copy<int2>(to, from, size_bytes / sizeof(int2));
        else
            _copy<int> (to, from, size_bytes / sizeof(int));
    }

    /**
     * Pack from channels to memory chunk (implementation)
     * Template parameter shiftmode governs shifting
     */
    template <ShiftMode shiftmode>
    inline __device__ void _packShift(int srcId, char *dstAddr, float3 shift) const
    {
        for (int i = 0; i < nChannels; i++)
        {
            const int size = channelSizes[i];
            const char *srcAddr = channelData[i] + size * srcId;

            _packShiftOneChannel<shiftmode>(srcAddr, dstAddr, size, i, shift);
            
            dstAddr += size;
        }
    }
    
    /**
     * Pack from local channels to dst channels (implementation)
     * Template parameter shiftmode governs shifting
     */
    template <ShiftMode shiftmode>
    inline __device__ void _packShift(int srcId, DevicePacker& dst, int dstId, float3 shift) const
    {
        assert (nChannels == dst.nChannels);
        
        for (int i = 0; i < nChannels; i++)
        {            
            const int size = channelSizes[i];
            const char *srcAddr =     channelData[i] + size * srcId;
            char       *dstAddr = dst.channelData[i] + size * dstId;

            _packShiftOneChannel<shiftmode>(srcAddr, dstAddr, size, i, shift);
        }
    }

    template <ShiftMode shiftmode>
    inline __device__ void _packShiftOneChannel(const char *srcAddr, char *dstAddr, int size, int channelId, float3 shift) const
    {
        int done = 0;

        if (shiftmode == ShiftMode::NeedShift)
        {
            if (channelShiftTypes[channelId] == sizeof(float))
            {
                float4 val = *((float4*) ( srcAddr ));
                val.x += shift.x;
                val.y += shift.y;
                val.z += shift.z;
                *((float4*) dstAddr) = val;

                done = sizeof(float4);
            }
            else if (channelShiftTypes[channelId] == sizeof(double))
            {
                double4 val = *((double4*) ( srcAddr ));
                val.x += shift.x;
                val.y += shift.y;
                val.z += shift.z;
                *((double4*) dstAddr) = val;

                done = sizeof(double4);
            }
        }

        copy(dstAddr + done, srcAddr + done, size - done);
    }
    
#endif /* __CUDACC__ */
    
protected:

    void registerChannel (DataManager& manager, int sz, char *ptr, int typesize, bool& needUpload, cudaStream_t stream);
    void registerChannels(PackPredicate predicate,
                          DataManager& manager, const std::string& pvName, bool& needUpload, cudaStream_t stream);
    void setAndUploadData(DataManager& manager, bool needUpload, cudaStream_t stream);
};
