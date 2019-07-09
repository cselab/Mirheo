#include "stream_pool.h"

#include <core/logger.h>

StreamPool::StreamPool(int n) :
    streams(n),
    eventsEnd(n)
{
    for (auto& s : streams)
        CUDA_Check( cudaStreamCreate(&s) );

    for (auto& e : eventsEnd)
        CUDA_Check( cudaEventCreate(&e) );

    CUDA_Check( cudaEventCreate(&eventStart) );
}

StreamPool::~StreamPool()
{
    for (auto& s : streams)
        CUDA_Check( cudaStreamDestroy(s) );

    for (auto& e : eventsEnd)
        CUDA_Check( cudaEventDestroy(e) );

    CUDA_Check( cudaEventDestroy(eventStart) );
}

const cudaStream_t& StreamPool::get(int id) const
{
    return streams[id];
}

void StreamPool::setStart(cudaStream_t streamBefore)
{
    constexpr unsigned int flags = 0; // must be zero according to docs
    
    CUDA_Check( cudaEventRecord(eventStart, streamBefore) );
    
    for (auto s : streams)
        CUDA_Check( cudaStreamWaitEvent(s, eventStart, flags) );
}

void StreamPool::setEnd(cudaStream_t streamAfter)
{
    constexpr unsigned int flags = 0; // must be zero according to docs
    
    for (int i = 0; i < streams.size(); ++i)
    {
        CUDA_Check( cudaEventRecord(eventsEnd[i], streams[i]) );
        CUDA_Check( cudaStreamWaitEvent(streamAfter, eventsEnd[i], flags) );
    }
}


