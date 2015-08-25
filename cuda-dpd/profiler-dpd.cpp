/*
 *  profiler-dpd.cpp
 *  Part of CTC/cuda-dpd-sem/
 *
 *  Created and authored by Diego Rossinelli on 2014-07-17.
 *  Copyright 2015. All rights reserved.
 *
 *  Users are NOT authorized
 *  to employ the present software for their own publications
 *  before getting a written permission from the author of this file.
 */

#include <cstdlib>
#include <cstdio>
#include <unistd.h>

#include <cuda_profiler_api.h>

#include "profiler-dpd.h"

#define CUDA_CHECK(ans) do { cudaAssert((ans), __FILE__, __LINE__); } while(0)
inline void cudaAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess) 
    {
	fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
	sleep(5);
	if (abort) exit(code);
    }
}

ProfilerDPD * ProfilerDPD::st = NULL;

ProfilerDPD& ProfilerDPD::singletone()
{
    if (st == NULL)
	return *(st = new ProfilerDPD);

    return *st;
}

void ProfilerDPD::start()
{
#ifdef _PROFILE_
    //CUDA_CHECK(cudaProfilerStart());
#endif
	
    //  CUDA_CHECK(cudaThreadSynchronize());
    CUDA_CHECK(cudaEventRecord(evstart));
}

ProfilerDPD::ProfilerDPD(): count(0), tf(0)
{
    CUDA_CHECK(cudaEventCreate(&evstart));
    CUDA_CHECK(cudaEventCreate(&evforce));
    
    _flush(true);
}

ProfilerDPD::~ProfilerDPD()
{
    CUDA_CHECK(cudaEventDestroy(evstart));
    CUDA_CHECK(cudaEventDestroy(evforce));
}

void ProfilerDPD::_flush(bool init)
{
    FILE * f = fopen("profdpd.txt", init ? "w" : "a");

    if (init)
	fprintf(f, "STEP ID\tFORCECOMP[s]\tREDUCE[s]\n");
    
    for(int i = 0; i < tfs.size(); ++i)
	fprintf(f, "%d\t%e\n", i + count, 1e-3 * tfs[i]);

    tfs.clear();
        
    fclose(f);
}
    
void ProfilerDPD::report()
{
    CUDA_CHECK(cudaEventSynchronize(evforce));

#ifdef _PROFILE_
    //CUDA_CHECK(cudaProfilerStop());
#endif
	
    float tforce;
    CUDA_CHECK(cudaEventElapsedTime(&tforce, evstart, evforce));
    	    
    tf += tforce;
    tfs.push_back(tforce);
    
    count++;
	    
    if (count % 100 == 0)
    {
	printf("dpd-profiler: force: %.2f ms\n", tf/count);
	_flush();
    }
}
