#include <cstdio>
#include <unistd.h>

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

ProfilerDPD::ProfilerDPD(): count(0), tf(0), tr(0), tt(0)
{
    CUDA_CHECK(cudaEventCreate(&evstart));
    CUDA_CHECK(cudaEventCreate(&evforce));
    CUDA_CHECK(cudaEventCreate(&evreduce));

    _flush(true);
}

ProfilerDPD::~ProfilerDPD()
{
    CUDA_CHECK(cudaEventDestroy(evstart));
    CUDA_CHECK(cudaEventDestroy(evforce));
    CUDA_CHECK(cudaEventDestroy(evreduce));
}

void ProfilerDPD::_flush(bool init)
{
    FILE * f = fopen("profdpd.txt", init ? "w" : "a");

    if (init)
	fprintf(f, "STEP ID\tFORCECOMP[s]\tREDUCE[s]\n");
    
    for(int i = 0; i < tfs.size(); ++i)
	fprintf(f, "%d\t%e\t%e\n", i + count, 1e-3 * tfs[i], 1e-3 * trs[i]);

    tfs.clear();
    trs.clear();
    
    fclose(f);
}
    
void ProfilerDPD::report()
{
    CUDA_CHECK(cudaEventSynchronize(evreduce));
    float tforce, treduce, ttotal;
    CUDA_CHECK(cudaEventElapsedTime(&tforce, evstart, evforce));
    CUDA_CHECK(cudaEventElapsedTime(&treduce, evforce, evreduce));
    CUDA_CHECK(cudaEventElapsedTime(&ttotal, evstart, evreduce));
	    
    tf += tforce;
    tr += treduce;
    tt += ttotal;

    tfs.push_back(tforce);
    trs.push_back(treduce);
    
    count++;
	    
    if (count % 100 == 0)
    {
	printf("times: %.2f ms %.2f ms -> F %.1f%%\n", tf/count, tr/count, tf/tt * 100);
	_flush();
    }
}