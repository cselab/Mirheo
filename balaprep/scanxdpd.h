typedef struct {
        int g_block_id;
        int g_blockcnt;
        int sum;
} sblockds_t;

#define MAXTHREADS 1024

__global__ void exclscn2e(int *d_data0, int *d_output0,
                        int *d_data1, int *d_output1,
                        int *d_data2, int *d_output2,
                        int *d_data3, int *d_output3,
                        int *d_data4, int *d_output4,
                        int *d_data5, int *d_output5,
                        int *d_data6, int *d_output6,
                        int *d_data7, int *d_output7) {
        const int twid=threadIdx.x%32;
        int wid=threadIdx.x/32;
        switch(wid) {
          case 0:
                if(twid<2) {
                        d_output0[twid]=d_data0[0]*twid;
                }
          return;
          case 1:
                if(twid<2) {
                        d_output1[twid]=d_data1[0]*twid;
                }
          return;
          case 2:
                if(twid<2) {
                        d_output2[twid]=d_data2[0]*twid;
                }
          return;
          case 3:
                if(twid<2) {
                        d_output3[twid]=d_data3[0]*twid;
                }
          return;
          case 4:
                if(twid<2) {
                        d_output4[twid]=d_data4[0]*twid;
                }
          return;
          case 5:
                if(twid<2) {
                        d_output5[twid]=d_data5[0]*twid;
                }
          return;
          case 6:
                if(twid<2) {
                        d_output6[twid]=d_data6[0]*twid;
                }
          return;
          case 7:
                if(twid<2) {
                        d_output7[twid]=d_data7[0]*twid;
                }
          return;
        }
}

__global__ void exclscnmb2e(int *d_data0, int *d_output0,
                        int *d_data1, int *d_output1,
                        int *d_data2, int *d_output2,
                        int *d_data3, int *d_output3,
                        int *d_data4, int *d_output4,
                        int *d_data5, int *d_output5,
                        int *d_data6, int *d_output6,
                        int *d_data7, int *d_output7) {
        const int twid=threadIdx.x;
        switch(blockIdx.x) {
          case 0:
                if(twid<2) {
                        d_output0[twid]=d_data0[0]*twid;
                }
          return;
          case 1:
                if(twid<2) {
                        d_output1[twid]=d_data1[0]*twid;
                }
          return;
          case 2:
                if(twid<2) {
                        d_output2[twid]=d_data2[0]*twid;
                }
          return;
          case 3:
                if(twid<2) {
                        d_output3[twid]=d_data3[0]*twid;
                }
          return;
          case 4:
                if(twid<2) {
                        d_output4[twid]=d_data4[0]*twid;
                }
          return;
          case 5:
                if(twid<2) {
                        d_output5[twid]=d_data5[0]*twid;
                }
          return;
          case 6:
                if(twid<2) {
                        d_output6[twid]=d_data6[0]*twid;
                }
          return;
          case 7:
                if(twid<2) {
                        d_output7[twid]=d_data7[0]*twid;
                }
          return;
        }
}

__global__ void exclscn2w(int *d_data, int *d_output, int size) {
  __shared__ int temp[32];
  int temp1, temp2, temp4;
  if(blockDim.x>MAXTHREADS) {
        printf("Invalid number of threads per block: %d, must be <=%d\n",blockDim.x,MAXTHREADS);
  }
  const int tid = threadIdx.x;
  temp4 = temp1 = (tid+blockIdx.x*blockDim.x<size)?d_data[tid+blockIdx.x*blockDim.x]:0;
  for (int d=1; d<32; d<<=1) {
         temp2 = __shfl_up(temp1,d);
        if (tid%32 >= d) temp1 += temp2;
  }
  if (tid%32 == 31) temp[tid/32] = temp1;
  __syncthreads();
  if (tid >= 32) { temp1 += temp[0]; }
  if(tid+blockIdx.x*blockDim.x<size) {
        d_output[tid+blockIdx.x*blockDim.x]=temp1-temp4;
  }
}

__global__ void exclscnmb2w(int *d_data0, int *d_output0,
                                int *d_data1, int *d_output1,
                                int *d_data2, int *d_output2,
                                int *d_data3, int *d_output3,
                                int *d_data4, int *d_output4,
                                int *d_data5, int *d_output5,
                                int *d_data6, int *d_output6,
                                int *d_data7, int *d_output7,
                                int *d_data8, int *d_output8,
                                int *d_data9, int *d_output9,
                                int *d_data10, int *d_output10,
                                int *d_data11, int *d_output11,
                                int size) {
  __shared__ int temp[32];
  int temp1, temp2, temp4;
  if(blockDim.x>MAXTHREADS) {
        printf("Invalid number of threads per block: %d, must be <=%d\n",blockDim.x,MAXTHREADS);
  }
  const int tid = threadIdx.x;
  switch(blockIdx.x) {
    case 0:
      temp4 = temp1 = (tid<size)?d_data0[tid]:0;
    break;
    case 1:
      temp4 = temp1 = (tid<size)?d_data1[tid]:0;
    break;
    case 2:
      temp4 = temp1 = (tid<size)?d_data2[tid]:0;
    break;
    case 3:
      temp4 = temp1 = (tid<size)?d_data3[tid]:0;
    break;
    case 4:
      temp4 = temp1 = (tid<size)?d_data4[tid]:0;
    break;
    case 5:
      temp4 = temp1 = (tid<size)?d_data5[tid]:0;
    break;
    case 6:
      temp4 = temp1 = (tid<size)?d_data6[tid]:0;
    break;
    case 7:
      temp4 = temp1 = (tid<size)?d_data7[tid]:0;
    break;
    case 8:
      temp4 = temp1 = (tid<size)?d_data8[tid]:0;
    break;
    case 9:
      temp4 = temp1 = (tid<size)?d_data9[tid]:0;
    break;
    case 10:
      temp4 = temp1 = (tid<size)?d_data10[tid]:0;
    break;
    case 11:
      temp4 = temp1 = (tid<size)?d_data11[tid]:0;
    break;
  }
  for (int d=1; d<32; d<<=1) {
         temp2 = __shfl_up(temp1,d);
         if (tid%32 >= d) temp1 += temp2;
  }
  if (tid%32 == 31) temp[tid/32] = temp1;
  __syncthreads();
  if (tid >= 32) { temp1 += temp[0]; }
  if(tid<size) {
    switch(blockIdx.x) {
    case 0:
      d_output0[tid]=temp1-temp4;
    break;
    case 1:
      d_output1[tid]=temp1-temp4;
    break;
    case 2:
      d_output2[tid]=temp1-temp4;
    break;
    case 3:
      d_output3[tid]=temp1-temp4;
    break;
    case 4:
      d_output4[tid]=temp1-temp4;
    break;
    case 5:
      d_output5[tid]=temp1-temp4;
    break;
    case 6:
      d_output6[tid]=temp1-temp4;
    break;
    case 7:
      d_output7[tid]=temp1-temp4;
    break;
    case 8:
      d_output8[tid]=temp1-temp4;
    break;
    case 9:
      d_output9[tid]=temp1-temp4;
    break;
    case 10:
      d_output10[tid]=temp1-temp4;
    break;
    case 11:
      d_output11[tid]=temp1-temp4;
    break;
    }
  }
}

__global__ void exclscnmb2ew(int *d_data0, int *d_output0,
                                int *d_data1, int *d_output1,
                                int *d_data2, int *d_output2,
                                int *d_data3, int *d_output3,
                                int *d_data4, int *d_output4,
                                int *d_data5, int *d_output5,
                                int *d_data6, int *d_output6,
                                int *d_data7, int *d_output7,
                                int *d_data8, int *d_output8,
                                int *d_data9, int *d_output9,
                                int *d_data10, int *d_output10,
                                int *d_data11, int *d_output11,
                                int *d_data20, int *d_output20,
                                int *d_data21, int *d_output21,
                                int *d_data22, int *d_output22,
                                int *d_data23, int *d_output23,
                                int *d_data24, int *d_output24,
                                int *d_data25, int *d_output25,
                                int *d_data26, int *d_output26,
                                int *d_data27, int *d_output27,
                                int size) {
  __shared__ int temp[32];
  int temp1, temp2, temp4;
  if(blockDim.x>MAXTHREADS) {
        printf("Invalid number of threads per block: %d, must be <=%d\n",blockDim.x,MAXTHREADS);
  }
  const int tid = threadIdx.x;
  switch(blockIdx.x) {
    case 0:
      temp4 = temp1 = (tid<size)?d_data0[tid]:0;
    break;
    case 1:
      temp4 = temp1 = (tid<size)?d_data1[tid]:0;
    break;
    case 2:
      temp4 = temp1 = (tid<size)?d_data2[tid]:0;
    break;
    case 3:
      temp4 = temp1 = (tid<size)?d_data3[tid]:0;
    break;
    case 4:
      temp4 = temp1 = (tid<size)?d_data4[tid]:0;
    break;
    case 5:
      temp4 = temp1 = (tid<size)?d_data5[tid]:0;
    break;
    case 6:
      temp4 = temp1 = (tid<size)?d_data6[tid]:0;
    break;
    case 7:
      temp4 = temp1 = (tid<size)?d_data7[tid]:0;
    break;
    case 8:
      temp4 = temp1 = (tid<size)?d_data8[tid]:0;
    break;
    case 9:
      temp4 = temp1 = (tid<size)?d_data9[tid]:0;
    break;
    case 10:
      temp4 = temp1 = (tid<size)?d_data10[tid]:0;
    break;
    case 11:
      temp4 = temp1 = (tid<size)?d_data11[tid]:0;
    break;
  }
  for (int d=1; d<32; d<<=1) {
         temp2 = __shfl_up(temp1,d);
         if (tid%32 >= d) temp1 += temp2;
  }
  if (tid%32 == 31) temp[tid/32] = temp1;
  __syncthreads();
  if (tid >= 32) { temp1 += temp[0]; }
  if(tid<size) {
    switch(blockIdx.x) {
    case 0:
      d_output0[tid]=temp1-temp4;
      if(tid<2) {
          d_output20[tid]=d_data20[0]*tid;
      }
    break;
    case 1:
      d_output1[tid]=temp1-temp4;
      if(tid<2) {
          d_output21[tid]=d_data21[0]*tid;
      }
    break;
    case 2:
      d_output2[tid]=temp1-temp4;
      if(tid<2) {
          d_output22[tid]=d_data22[0]*tid;
      }
    break;
    case 3:
      d_output3[tid]=temp1-temp4;
      if(tid<2) {
          d_output23[tid]=d_data23[0]*tid;
      }
    break;
    case 4:
      d_output4[tid]=temp1-temp4;
      if(tid<2) {
          d_output24[tid]=d_data24[0]*tid;
      }
    break;
    case 5:
      d_output5[tid]=temp1-temp4;
      if(tid<2) {
          d_output25[tid]=d_data25[0]*tid;
      }
    break;
    case 6:
      d_output6[tid]=temp1-temp4;
      if(tid<2) {
          d_output26[tid]=d_data26[0]*tid;
      }
    break;
    case 7:
      d_output7[tid]=temp1-temp4;
      if(tid<2) {
          d_output27[tid]=d_data27[0]*tid;
      }
    break;
    case 8:
      d_output8[tid]=temp1-temp4;
    break;
    case 9:
      d_output9[tid]=temp1-temp4;
    break;
    case 10:
      d_output10[tid]=temp1-temp4;
    break;
    case 11:
      d_output11[tid]=temp1-temp4;
    break;
    }
  }
}


__global__ void exclscnmb(int *d_data, int *d_output, int size) {
  __shared__ int temp[32];
  int temp1, temp2, temp3, temp4;
  if(blockDim.x>MAXTHREADS) {
        printf("Invalid number of threads per block: %d, must be <=%d\n",blockDim.x,MAXTHREADS);
  }
  int tid = threadIdx.x;
  temp4 = temp1 = (tid+blockIdx.x*blockDim.x<size)?d_data[tid+blockIdx.x*blockDim.x]:0;
  for (int d=1; d<32; d<<=1) {
         temp2 = __shfl_up(temp1,d);
        if (tid%32 >= d) temp1 += temp2;
  }
  if (tid%32 == 31) temp[tid/32] = temp1;
  __syncthreads();
  if (threadIdx.x < 32) {
        temp2 = 0;
        if (tid < blockDim.x/32) {
                temp2 = temp[threadIdx.x];
        }
        for (int d=1; d<32; d<<=1) {
         temp3 = __shfl_up(temp2,d);
         if (tid%32 >= d) {temp2 += temp3;}
        }
        if (tid < blockDim.x/32) { temp[tid] = temp2; }
  }
  __syncthreads();
  if (tid >= 32) { temp1 += temp[tid/32 - 1]; }
  __syncthreads();
  if(tid+blockIdx.x*blockDim.x<size) {
        d_output[tid+blockIdx.x*blockDim.x]=temp1-temp4;
  }
}

__global__ void exclscan(int *d_data, int *d_output, int size, sblockds_t *ptoblockds) {
  __shared__ int temp[32];
  __shared__ unsigned int my_blockId;
  int temp1, temp2, temp3, temp4;
  if(blockDim.x>MAXTHREADS) {
        printf("Invalid number of threads per block: %d, must be <=%d\n",blockDim.x,MAXTHREADS);
  }
  if (threadIdx.x==0) {
         my_blockId = atomicAdd( &(ptoblockds->g_block_id), 1 );
  }
  __syncthreads();
  int tid = threadIdx.x;
  temp4 = temp1 = (tid+my_blockId*blockDim.x<size)?d_data[tid+my_blockId*blockDim.x]:0;
  for (int d=1; d<32; d<<=1) {
         temp2 = __shfl_up(temp1,d);
        if (tid%32 >= d) temp1 += temp2;
  }
  if (tid%32 == 31) temp[tid/32] = temp1;
  __syncthreads();
  if (threadIdx.x < 32) {
        temp2 = 0;
        if (tid < blockDim.x/32) {
                temp2 = temp[threadIdx.x];
        }
        for (int d=1; d<32; d<<=1) {
         temp3 = __shfl_up(temp2,d);
         if (tid%32 >= d) {temp2 += temp3;}
        }
        if (tid < blockDim.x/32) { temp[tid] = temp2; }
  }
  __syncthreads();
  if (tid >= 32) { temp1 += temp[tid/32 - 1]; }
  __syncthreads();
  if (threadIdx.x==(blockDim.x-1)) {
        do {} while( atomicAdd(&(ptoblockds->g_blockcnt),0) < my_blockId );
        temp[0]=ptoblockds->sum;
        if(my_blockId==(gridDim.x-1)) { /* it is the last block; reset for next iteration */
                ptoblockds->sum=0;
                ptoblockds->g_blockcnt=0;
                ptoblockds->g_block_id=0;
        } else {
                ptoblockds->sum=temp[0]+temp1;
                atomicAdd(&(ptoblockds->g_blockcnt),1);
        }
        __threadfence();  // wait for write completion
  }
  __syncthreads();
  temp1+=temp[0];
  if(tid+my_blockId*blockDim.x<size) {
        d_output[tid+my_blockId*blockDim.x]=temp1-temp4;
  }
}

#undef MAXTHREADS
