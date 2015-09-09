/*
 *  minmax.cu
 *  Part of uDeviceX/balaprep/
 *
 *  Created and authored by Massimo Bernaschi on 2015-03-13.
 *  Copyright 2015. All rights reserved.
 *
 *  Users are NOT authorized
 *  to employ the present software for their own publications
 *  before getting a written permission from the author of this file.
 */

#include <stdio.h>
#include <stdlib.h>
#include <thrust/scan.h>
#define NDIR 1
#define SIZE1  500
#define SIZE2   49
#define SIZE3    2
#define N1  1
#define N2  0
#define N3  0
//#define N3  8
#define MY_CUDA_CHECK( call) {                                    \
    cudaError err = call;                                                    \
    if( cudaSuccess != err) {                                                \
        fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",        \
                __FILE__, __LINE__, cudaGetErrorString( err) );              \
        exit(EXIT_FAILURE);                                                  \
    } }
#define NTHREADS 1024

/* following definition is fake! */
struct Particle {
       float x[3]; float u[3];
};

typedef struct {
        int g_block_id;
        int g_blockcnt;
        float3 minval;
        float3 maxval;
} sblockds_t;

#define MAXV     100000000.
#define MINV    -100000000.

void maxminCTC(struct Particle *rbc, int size, int n, float3 *minrbc, float3 *maxrbc);
void maxminRBC(struct Particle *rbc, int size, int n, float3 *minrbc, float3 *maxrbc);

__global__ void minshuffle(float3 *, float3 *, float3 *, int, sblockds_t *);

main(int argc, char *argv[]) {
         int ntimes=1;
         struct Particle *data[NDIR];
         float3 *outputmin[NDIR], *outputmax[NDIR];;
         int size[NDIR];
         struct Particle *d_data[NDIR];
         float3 *d_outputmin[NDIR], *d_outputmax[NDIR];
         sblockds_t *ptoblockds;
         cudaEvent_t start, stop;
         cudaEventCreate(&start);
         cudaEventCreate(&stop);
         if(argc>1) {
                    ntimes=atoi(argv[1]);
                    if(ntimes<1) {
                       fprintf(stderr,"Invalid number of repetitions %d\n",ntimes);
                       exit(1);
                    }
         }
         sblockds_t h_ptoblockds;
         h_ptoblockds.g_block_id=0;
         h_ptoblockds.g_blockcnt=0;
         h_ptoblockds.minval.x=MAXV;
         h_ptoblockds.maxval.x=MINV;
         h_ptoblockds.minval.y=MAXV;
         h_ptoblockds.maxval.y=MINV;
         h_ptoblockds.minval.z=MAXV;
         h_ptoblockds.maxval.z=MINV;
         MY_CUDA_CHECK(cudaMalloc((void **)&ptoblockds,sizeof(sblockds_t)*N1));
         MY_CUDA_CHECK(cudaMemcpy(ptoblockds,&h_ptoblockds,sizeof(sblockds_t)*N1,cudaMemcpyHostToDevice));

         for(int i=0; i<N1; i++) {
                 data[i]= new struct Particle[SIZE1*ntimes];
                 MY_CUDA_CHECK(cudaMalloc((void **)&d_data[i],ntimes*SIZE1*sizeof(struct Particle)));
                 outputmin[i]= new float3[ntimes];
                 outputmax[i]= new float3[ntimes];
                 size[i]=SIZE1;
                 MY_CUDA_CHECK(cudaMalloc((void **)&d_outputmin[i],ntimes*sizeof(float3)));
                 MY_CUDA_CHECK(cudaMalloc((void **)&d_outputmax[i],ntimes*sizeof(float3)));
                 for(int j=0; j<SIZE1*ntimes; j++) {
                     data[i][j].x[0]=(float)(drand48()*100);
                     data[i][j].x[1]=(float)(drand48()*33);
                     data[i][j].x[2]=(float) (drand48()*67);
                 }
                 MY_CUDA_CHECK( cudaMemcpy( d_data[i], data[i], ntimes*SIZE1*sizeof(struct Particle),
                                            cudaMemcpyHostToDevice ) );

         }
         cudaEventRecord(start);

         for(int i=0; i<NDIR; i++) {
             if(size[i] > NTHREADS) {
//                  minmaxmb<<<(size[i]+NTHREADS-1)/NTHREADS,NTHREADS>>>(d_data[i], d_outputmin[i], d_outputmax[i],size[i],ptoblockds);
                  maxminCTC(d_data[i], size[i], ntimes, d_outputmin[i], d_outputmax[i]);
             } else {
//                  minmaxob<<<ntimes,((size[i]+31)/32)*32>>>(d_data[i], d_outputmin[0], d_outputmax[0], size[i]);
                  maxminRBC(d_data[i], size[i], ntimes, d_outputmin[0], d_outputmax[0]);
             }
             MY_CUDA_CHECK( cudaMemcpy( outputmin[i], d_outputmin[0], ntimes*sizeof(float3),
                                            cudaMemcpyDeviceToHost ) );
             MY_CUDA_CHECK( cudaMemcpy( outputmax[i], d_outputmax[0], ntimes*sizeof(float3),
                                            cudaMemcpyDeviceToHost ) );


         }
         cudaEventRecord(stop);
         cudaEventSynchronize(stop);
         float milliseconds = 0;
         cudaEventElapsedTime(&milliseconds, start, stop);
         printf("Time=%f milliseconds\n",milliseconds);
         float3 h_min, h_max;
         for(int l=0; l<ntimes; l++) {
         h_min.x=MAXV;
         h_min.y=MAXV;
         h_min.z=MAXV;
         h_max.x=MINV;
         h_max.y=MINV;
         h_max.z=MINV;
         for(int i=0; i<NDIR; i++) {
                 for(int j=size[i]*l; j<size[i]*(l+1); j++) {
                         h_min.x=(data[i][j].x[0]<h_min.x)?data[i][j].x[0]:h_min.x;
                         h_max.x=(data[i][j].x[0]>h_max.x)?data[i][j].x[0]:h_max.x;
                         h_min.y=(data[i][j].x[1]<h_min.y)?data[i][j].x[1]:h_min.y;
                         h_max.y=(data[i][j].x[1]>h_max.y)?data[i][j].x[1]:h_max.y;
                         h_min.z=(data[i][j].x[2]<h_min.z)?data[i][j].x[2]:h_min.z;
                         h_max.z=(data[i][j].x[2]>h_max.z)?data[i][j].x[2]:h_max.z;
                 }
         }
         printf("host min x=%f, host max x=%f, gpu min x=%f, gpu max x=%f\n",
                 h_min.x,h_max.x,outputmin[0][l].x,outputmax[0][l].x);
         printf("host min y=%f, host max y=%f, gpu min y=%f, gpu max y=%f\n",
                 h_min.y,h_max.y,outputmin[0][l].y,outputmax[0][l].y);
         printf("host min z=%f, host max z=%f, gpu min z=%f, gpu max z=%f\n",
                 h_min.z,h_max.z,outputmin[0][l].z,outputmax[0][l].z);
         }
         for(int i=0; i<NDIR; i++) {
                  delete data[i];
                  delete outputmin[i];
                  delete outputmax[i];
         }
}
