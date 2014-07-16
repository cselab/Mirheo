#pragma once

#include <vector>

class ProfilerDPD
{
    int count;
    float tf, tr, tt;
    cudaEvent_t evstart, evforce, evreduce;
    std::vector<double> tfs, trs;

    void _flush(bool init = false);
     
public:
    
    ProfilerDPD();
    ~ProfilerDPD();

    void start() { cudaEventRecord(evstart); }
    void force() { cudaEventRecord(evforce); }
    void reduce() { cudaEventRecord(evreduce); }
    void report();
} ;
