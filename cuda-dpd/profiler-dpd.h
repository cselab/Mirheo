/*
 *  profiler-dpd.h
 *  Part of uDeviceX/cuda-dpd-sem/
 *
 *  Created and authored by Diego Rossinelli on 2014-07-23.
 *  Copyright 2015. All rights reserved.
 *
 *  Users are NOT authorized
 *  to employ the present software for their own publications
 *  before getting a written permission from the author of this file.
 */

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

    ProfilerDPD();
    ~ProfilerDPD();
    
public:

    static ProfilerDPD * st;
    
    static ProfilerDPD& singletone();

    void start();
    void force() { cudaEventRecord(evforce); }
    void report();
};

