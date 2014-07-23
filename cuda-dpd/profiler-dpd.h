#pragma once

#include <cuda_runtime.h>
#include <vector>

class ProfilerDPD
{
    int count;
    float tf;
    cudaEvent_t evstart, evforce;
    std::vector<double> tfs;

    void _flush(bool init = false);
    
public:
    
    ProfilerDPD();
    ~ProfilerDPD();

    void start();
    void force() { cudaEventRecord(evforce); }
    void report();
};

