#pragma once

#include <cuda_runtime.h>
#include <vector>

class StreamPool
{
public:
    StreamPool(size_t n);
    ~StreamPool();

    /*
     * Set all streams to follow `streamBefore` asynchronously
     * If the streams are not yet created, will create them with 
     * same priority as `streamBefore`.
     */
    void setStart(cudaStream_t streamBefore);

    /*
     * Get ith stream
     * Must be called after `setStart`
     */
    const cudaStream_t& get(int id) const;

    /*
     * Set `streamAfter` to follow all streams asynchronously
     * Must be called after `setStart`
     */
    void setEnd(cudaStream_t streamAfter);
    
private:
    size_t n;
    std::vector<cudaStream_t> streams;
    std::vector<cudaEvent_t> eventsEnd;
    cudaEvent_t eventStart;
};
