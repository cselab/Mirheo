#pragma once

#include <mirheo/core/utils/cpu_gpu_defines.h>

#include <cstdint>

namespace mirheo
{

/** \brief A structure that holds a pair of buffer index and destination index in a single \c int32_t.

    This is used to create a packing map; each entity to pack needs to know in which buffer to go (bufId)
    and where in that buffer (id).
 */
class __align__(4) MapEntry
{
 public:
    __HD__ MapEntry() = delete;
    /** Create a map entry
        \param id The destination index in the dest buffer
        \param bufId The destination buffer index
     */
    __HD__ MapEntry(uint32_t id, uint32_t bufId) :
        i_((maskId & id) | (bufId << bufShift))
    {}

    /// \return the index within the buffer
    __HD__ inline uint32_t getId()    const {return i_ & maskId;}
    /// \return the buffer index
    __HD__ inline uint32_t getBufId() const {return i_ >> bufShift;}

    /// Set the index within the buffer
    __HD__ inline void setId(uint32_t id)
    {
        uint32_t bufInfo = maskBuf & i_;
        i_ = bufInfo | id;
    }

    /// Set the buffer index
    __HD__ inline void setBufId(uint32_t bufId)
    {
        uint32_t  idInfo = maskId & i_;
        uint32_t bufInfo = bufId << bufShift;
        i_ = bufInfo | idInfo;
    }

 private:

    // 27 < 2^5 buffers max
    static constexpr int bufWidth = 5;
    static constexpr int bufShift = 32 - bufWidth;
    static constexpr uint32_t maskAll = 0xffffffff;
    static constexpr uint32_t maskId  = (maskAll << bufWidth) >> bufWidth;
    static constexpr uint32_t maskBuf = ~maskId;

    uint32_t i_ {0};
};

/** \brief Assign each thread to a buffer so that all threads work on one element.
    \param nBuffers The total number of buffers
    \param offsets The starts (in number of elements) of each buffer
    \param tid The thread index within the block
    \return The buffer index that the thread will work with
 */
inline __HD__ int dispatchThreadsPerBuffer(int nBuffers, const int *offsets, int tid)
{
    int low = 0, hig = nBuffers;
    while (hig > low+1)
    {
        int mid = (low + hig) / 2;
        bool moveUp = tid >= offsets[mid];
        low = moveUp ? mid : low;
        hig = moveUp ? hig : mid;
    }
    return low;
}

} // namespace mirheo
