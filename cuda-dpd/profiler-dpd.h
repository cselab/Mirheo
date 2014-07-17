#pragma once

#include <cuda_runtime.h>
#include <vector>

class ProfilerDPD
{
    bool nvprof;
    int count;
    float tf, tr, tt;
    cudaEvent_t evstart, evforce, evreduce;
    std::vector<double> tfs, trs;

    void _flush(bool init = false);
    
public:
    
    ProfilerDPD(bool nvprof);
    ~ProfilerDPD();

    void start();
    void force() { cudaEventRecord(evforce); }
    void reduce() { cudaEventRecord(evreduce); }
    void report();
} ;
