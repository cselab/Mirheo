#pragma once

#include <map>

#include "common.h"

class ScanEngine
{
    std::map<cudaStream_t, SimpleDeviceBuffer<uint> *> str2buf;
    
public:
    void exclusive(cudaStream_t stream, uint *d_Dst, uint *d_Src, uint arrayLength);
   
    ~ScanEngine();
};

