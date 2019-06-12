#pragma once

#include "../data_manager.h"

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
    char **channelData       = nullptr;   ///< device pointers of the packed data

#ifdef __CUDACC__
    /**
     * Pack entity with id srcId into memory starting with dstAddr
     * with no shift
     */
    inline __device__ void pack(int srcId, char *dstAddr) const
    {
        for (int i = 0; i < nChannels; i++)
        {
            const int size = channelSizes[i];
            const char *srcAddr = channelData[i] + size * srcId;

            copy(dstAddr, srcAddr, size);
            
            dstAddr += size;
        }
    }

    /**
     * Pack entity with id srcId into another packer at id dstId
     * with no shift
     * Assumes that the 2 packers contain the same channels
     */
    inline __device__ void pack(int srcId, DevicePacker& dst, int dstId) const
    {
        assert (nChannels == dst.nChannels);
        
        for (int i = 0; i < nChannels; i++)
        {            
            const int size = channelSizes[i];
            const char *srcAddr =     channelData[i] + size * srcId;
            char       *dstAddr = dst.channelData[i] + size * dstId;

            copy(dstAddr, srcAddr, size);
        }
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


#endif /* __CUDACC__ */
    
private:

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
    
#endif /* __CUDACC__ */
    
protected:

    void registerChannel (DataManager& manager, int sz, char *ptr, bool& needUpload, cudaStream_t stream);
    void registerChannels(PackPredicate predicate, DataManager& manager, const std::string& pvName, bool& needUpload, cudaStream_t stream);
    void setAndUploadData(DataManager& manager, bool needUpload, cudaStream_t stream);
};
