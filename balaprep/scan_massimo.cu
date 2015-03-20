/*
 *  scan_massimo.cu
 *  Part of CTC/balaprep/
 *
 *  Created and authored by Massimo Bernaschi on 2015-03-09.
 *  Copyright 2015. All rights reserved.
 *
 *  Users are NOT authorized
 *  to employ the present software for their own publications
 *  before getting a written permission from the author of this file.
 */

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
//    static  sblockds_t *ptoblockds;

    int mb[6], mw[12], maxscan=0;
    static int **d_aopd, **d_aopr;
    static int *d_sizescan;
    int h_sizescan[18];
    const int *h_aopd[26], *h_aopr[26];
    if(newscani) {
//        CUDA_CHECK(cudaMalloc((void **)&ptoblockds,6*sizeof(sblockds_t)));
//        CUDA_CHECK(cudaMemset(ptoblockds,0,6*sizeof(sblockds_t)));
        CUDA_CHECK(cudaMalloc((void **)&d_sizescan,18*sizeof(int)));
        CUDA_CHECK(cudaMalloc((void ***)&d_aopd,sizeof(int *)*26));
        CUDA_CHECK(cudaMalloc((void ***)&d_aopr,sizeof(int *)*26));
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
                h_aopd[i]=count[mb[i]];
                h_aopr[i]=result[mb[i]];
                maxscan=(maxscan<h_sizescan[i])?h_sizescan[i]:maxscan;
        }
        for(int i = 0; i < 12; ++i) {
                h_sizescan[6+i]=sizes[mw[i]];
                h_aopd[6+i]=count[mw[i]];
                h_aopr[6+i]=result[mw[i]];
        }
        h_aopd[18]=count[0];
        h_aopr[18]=result[0];
        h_aopd[19]=count[1];
        h_aopr[19]=result[1];
        h_aopd[20]=count[3];
        h_aopr[20]=result[3];
        h_aopd[21]=count[4];
        h_aopr[21]=result[4];
        h_aopd[22]=count[9];
        h_aopr[22]=result[9];
        h_aopd[23]=count[10];
        h_aopr[23]=result[10];
        h_aopd[24]=count[12];
        h_aopr[24]=result[12];
        h_aopd[25]=count[13];
        h_aopr[25]=result[13];
        CUDA_CHECK(cudaMemcpy(d_sizescan,h_sizescan,18*sizeof(int),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK( cudaMemcpy( d_aopd, h_aopd, 26*sizeof(int *),
                                cudaMemcpyHostToDevice ) );
        CUDA_CHECK( cudaMemcpy( d_aopr, h_aopr, 26*sizeof(int *),
                                cudaMemcpyHostToDevice ) );

        newscani=0;
    }


#if defined(_TIME_PROFILE_)
   if (lit % 500 == 0)
        CUDA_CHECK(cudaEventRecord(evstart));
#endif
#define NTHREADS 1024
#if 0
        excl26scan<<<12+(6*((maxscan+NTHREADS-1)/NTHREADS)),NTHREADS,0,stream>>>(
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
#else
          excl26scanaopob<<<18,NTHREADS>>>(d_aopd, d_aopr, d_sizescan);
#endif
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
