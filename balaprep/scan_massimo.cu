#include "scanxdpd.h"
 
using namespace std;
void scan_massimo(const int * const count[26], int * const result[26], const int sizes[26], cudaStream_t stream)
{
    CUDA_CHECK(cudaPeekAtLastError());
#if defined(_TIME_PROFILE_)
    static  int lit=0;
    if(lit==0) {
        CUDA_CHECK(cudaEventCreate(&evstart));
        CUDA_CHECK(cudaEventCreate(&evstop));
    }
#endif
    static int newscani=1;
    static  sblockds_t *ptoblockds;
#endif
    static int mb[6], mw[12], maxscan=0;
    static int *d_sizescan;
    int h_sizescan[18];
    if(newscani) {
   	CUDA_CHECK(cudaMalloc((void **)&ptoblockds,6*sizeof(sblockds_t)));
        CUDA_CHECK(cudaMemset(ptoblockds,0,6*sizeof(sblockds_t)));
   	CUDA_CHECK(cudaMalloc((void **)&d_sizescan,18*sizeof(int)));
	mb[0]=8;
	mb[1]=17;
	mb[2]=20;
	mb[3]=23;
	mb[4]=24;
	mb[5]=25;
	mw[0]=2;
	mw[1]=5;
	mw[2]=6;
	mw[3]=7;
	mw[4]=11;
	mw[5]=14;
	mw[6]=15;
	mw[7]=16;
	mw[8]=18;
	mw[9]=19;
	mw[10]=21;
	mw[11]=22;
   	for(int i = 0; i < 6; ++i) {
		h_sizescan[i]=sizes[mb[i]];
		maxscan=(maxscan<h_sizescan[i])?h_sizescan[i]:maxscan;
    	}
   	for(int i = 0; i < 12; ++i) {
		h_sizescan[6+i]=sizes[mw[i]];
    	}
        CUDA_CHECK(cudaMemcpy(d_sizescan,h_sizescan,18*sizeof(int),
			      cudaMemcpyHostToDevice));
   	newscani=0;
    }

	
#if defined(_TIME_PROFILE_) 
   if (lit % 500 == 0)
        CUDA_CHECK(cudaEventRecord(evstart));
#endif
#define NTHREADS 1024
	excl26scan<<<12+(6*((maxscan+NTHREADS-1)/NTHREADS)),NTHREADS,0,streams>>>(
	  count[mb[0]],result[mb[0]],
	  count[mb[1]],result[mb[1]],
	  count[mb[2]],result[mb[2]],
	  count[mb[3]],result[mb[3]],
	  count[mb[4]],result[mb[4]],
	  count[mb[5]],result[mb[5]],
	  count[mw[0]],result[mw[0]],
	  count[mw[1]],result[mw[1]],
	  count[mw[2]],result[mw[2]],
	  count[mw[3]],result[mw[3]],
	  count[mw[4]],result[mw[4]],
	  count[mw[5]],result[mw[5]],
	  count[mw[6]],result[mw[6]],
	  count[mw[7]],result[mw[7]],
	  count[mw[8]],result[mw[8]],
	  count[mw[9]],result[mw[9]],
	  count[mw[10]],result[mw[10]],
	  count[mw[11]],result[mw[11]],
	  count[0],result[0],
	  count[1],result[1],
	  count[3],result[3],
	  count[4],result[4],
	  count[9],result[9],
	  count[10],result[10],
          count[12],result[12],
          count[13],result[13],
	  d_sizescan, maxscan, ptoblockds);
#if defined(_TIME_PROFILE_)
    if (lit % 500 == 0)
    {
        CUDA_CHECK(cudaEventRecord(evstop));
        CUDA_CHECK(cudaEventSynchronize(evstop));

        float tms;
        CUDA_CHECK(cudaEventElapsedTime(&tms, evstart, evstop));
        if(cntlwtimer<maxcntimer) {
                lwtimer[cntlwtimer++]=tms;
        }
    }
    lit++;
#endif
}
