#include <stdio.h>
#include "minmax.h"

#define MY_CUDA_CHECK( call) {                                    \
    cudaError err = call;                                                    \
    if( cudaSuccess != err) {                                                \
        fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",        \
                __FILE__, __LINE__, cudaGetErrorString( err) );              \
        exit(EXIT_FAILURE);                                                  \
    } }

void maxminRBC(struct Particle *rbc, int size, int n, float3 *minrbc, float3 *maxrbc) {
         minmaxob<<<n,((size+31)/32)*32>>>(rbc, minrbc, maxrbc, size);
/* n is the number of rbc, size is the number of points per rbc */
}

void maxminCTC(struct Particle *ctc, int size, int n, float3 *minctc, float3 *maxctc) {
        static int nctc=-1;
        static sblockds_t *ptoblockds=NULL;
        if(n>nctc) {
           sblockds_t *h_ptoblockds=new sblockds_t[n];
           for(int i=0; i<n; i++) {
                   h_ptoblockds[i].g_block_id=0;
                   h_ptoblockds[i].g_blockcnt=0;
                   h_ptoblockds[i].minval.x=MAXV;
                   h_ptoblockds[i].maxval.x=MINV;
                   h_ptoblockds[i].minval.y=MAXV;
                   h_ptoblockds[i].maxval.y=MINV;
                   h_ptoblockds[i].minval.z=MAXV;
                   h_ptoblockds[i].maxval.z=MINV;
           }
           if(ptoblockds!=NULL) { cudaFree(ptoblockds); };
           MY_CUDA_CHECK(cudaMalloc((void **)&ptoblockds,sizeof(sblockds_t)*n));
           MY_CUDA_CHECK(cudaMemcpy(ptoblockds,h_ptoblockds,sizeof(sblockds_t)*n,cudaMemcpyHostToDevice));
           delete h_ptoblockds;
        }

        int nblocks=n*((size+MAXTHREADS-1)/MAXTHREADS);
        minmaxmba<<<nblocks,MAXTHREADS>>>(ctc, minctc, maxctc, size, ptoblockds);
/* n is the number of ctc, size is the number of points per ctc */
}
