#pragma once

#include <cuda_runtime.h>
#include <vector>

class StreamPool
{
public:
    StreamPool(int n);
    ~StreamPool();

    const cudaStream_t& get(int id) const;
    void setStart(cudaStream_t streamBefore);
    void setEnd(cudaStream_t streamAfter);
    
private:
    std::vector<cudaStream_t> streams;
    std::vector<cudaEvent_t> eventsEnd;
    cudaEvent_t eventStart;
};
