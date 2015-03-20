/*
 *  scan.h
 *  Part of CTC/mpi-dpd/
 *
 *  Created and authored by Diego Rossinelli on 2014-11-28.
 *  Copyright 2015. All rights reserved.
 *
 *  Users are NOT authorized
 *  to employ the present software for their own publications
 *  before getting a written permission from the author of this file.
 */

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

