#include "minmax.h"

maxminRBC(struct Particle *rbc, int size, int n, float3 *minrbc, float3 *maxrbc) {
                 minmaxob<<<n,((size+31)/32)*32>>>(rbc, minrbc, maxrbc, size);
/* n is the number of rbc, size is the number of points per rbc */
}

maxminCTC(struct Particle *ctc, int size, int n, float3 *minctc, float3 *maxctc) {
        static int newscani=1;
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
           cudaFree(ptoblockds);
           MY_CUDA_CHECK(cudaMalloc((void **)&ptoblockds,sizeof(sblockds_t)*n));
           MY_CUDA_CHECK(cudaMemcpy(ptoblockds,h_ptoblockds,sizeof(sblockds_t)*n,cudaMemcpyHostToDevice));
           delete h_ptoblocksds;
        }

        int nblocks=n*((size+MAXTHREADS-1)/MAXTHREADS);
        minmaxmba<<<nblocks,MAXTHREADS>>>(ctc, minctc, maxctc, size, sblockds_t *ptoblockds);
/* n is the number of ctc, size is the number of points per ctc */
}
