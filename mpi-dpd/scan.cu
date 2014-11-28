#include "scan.h"

//THIS CODE WAS NAIVELY EXTENDED FROM THE CUDA SDK!!! SHAME ON ME.

//All three kernels run 512 threads per workgroup
//Must be a power of two
#define THREADBLOCK_SIZE 256

////////////////////////////////////////////////////////////////////////////////
// Basic ccan codelets
////////////////////////////////////////////////////////////////////////////////
//Naive inclusive scan: O(N * log2(N)) operations
//Allocate 2 * 'size' local memory, initialize the first half
//with 'size' zeros avoiding if(pos >= offset) condition evaluation
//and saving instructions
inline __device__ uint scan1Inclusive(uint idata, volatile uint *s_Data, uint size)
{
    uint pos = 2 * threadIdx.x - (threadIdx.x & (size - 1));
    s_Data[pos] = 0;
    pos += size;
    s_Data[pos] = idata;

    for (uint offset = 1; offset < size; offset <<= 1)
    {
        __syncthreads();
        uint t = s_Data[pos] + s_Data[pos - offset];
        __syncthreads();
        s_Data[pos] = t;
    }

    return s_Data[pos];
}

inline __device__ uint scan1Exclusive(uint idata, volatile uint *s_Data, uint size)
{
    return scan1Inclusive(idata, s_Data, size) - idata;
}

inline __device__ uint4 scan4Inclusive(uint4 idata4, volatile uint *s_Data, uint size)
{
    //Level-0 inclusive scan
    idata4.y += idata4.x;
    idata4.z += idata4.y;
    idata4.w += idata4.z;

    //Level-1 exclusive scan
    uint oval = scan1Exclusive(idata4.w, s_Data, size / 4);

    idata4.x += oval;
    idata4.y += oval;
    idata4.z += oval;
    idata4.w += oval;

    return idata4;
}

//Exclusive vector scan: the array to be scanned is stored
//in local thread memory scope as uint4
inline __device__ uint4 scan4Exclusive(uint4 idata4, volatile uint *s_Data, uint size)
{
    uint4 odata4 = scan4Inclusive(idata4, s_Data, size);
    odata4.x -= idata4.x;
    odata4.y -= idata4.y;
    odata4.z -= idata4.z;
    odata4.w -= idata4.w;
    return odata4;
}

////////////////////////////////////////////////////////////////////////////////
// Scan kernels
////////////////////////////////////////////////////////////////////////////////
__global__ void scanExclusiveShared(uint4 *d_Dst, uint4 *d_Src, uint size, const int arrayLength)
{
    __shared__ uint s_Data[2 * THREADBLOCK_SIZE];

    uint pos = blockIdx.x * blockDim.x + threadIdx.x;

    //Load data
    uint4 idata4 = make_uint4(0, 0, 0, 0);

    if (pos * 4 < arrayLength)
	idata4 = d_Src[pos];

    //Calculate exclusive scan
    uint4 odata4 = scan4Exclusive(idata4, s_Data, size);

    //Write back
    if (pos * 4 < arrayLength)
	d_Dst[pos] = odata4;
}

//Exclusive scan of top elements of bottom-level scans (4 * THREADBLOCK_SIZE)
__global__ void scanExclusiveShared2(uint *d_Buf, uint *d_Dst, uint *d_Src, uint N, uint arrayLength, uint originalArrayLength)
{
    __shared__ uint s_Data[2 * THREADBLOCK_SIZE];

    //Skip loads and stores for inactive threads of last threadblock (pos >= N)
    uint pos = blockIdx.x * blockDim.x + threadIdx.x;

    //Load top elements
    //Convert results of bottom-level scan back to inclusive
    uint idata = 0;
    const int entry = (4 * THREADBLOCK_SIZE) - 1 + (4 * THREADBLOCK_SIZE) * pos;
    const bool valid = pos < N && entry >= 0 && entry < originalArrayLength;
    
    if (valid)
	idata =
            d_Dst[entry] +
            d_Src[entry];
 
    //Compute
    uint odata = scan1Exclusive(idata, s_Data, arrayLength);
    
    //Avoid out-of-bound access
    if (pos < N)
    {
//	printf("gid %d my odata: %d, idata %d\n", pos, odata, idata);
        d_Buf[pos] = odata;
    }
}

//Final step of large-array scan: combine basic inclusive scan with exclusive scan of top elements of input arrays
__global__ void uniformUpdate(uint4 *d_Data, uint *d_Buffer, const int arrayLength)
{
    __shared__ uint buf;
    uint pos = blockIdx.x * blockDim.x + threadIdx.x;

    if (threadIdx.x == 0)
    {
        buf = d_Buffer[blockIdx.x];
    }

    __syncthreads();

    if (pos * 4 < arrayLength)
    {
	uint4 data4 = d_Data[pos];
	data4.x += buf;
	data4.y += buf;
	data4.z += buf;
	data4.w += buf;
	d_Data[pos] = data4;
    }
}

////////////////////////////////////////////////////////////////////////////////
// Interface function
////////////////////////////////////////////////////////////////////////////////
//Derived as 32768 (max power-of-two gridDim.x) * 4 * THREADBLOCK_SIZE
//Due to scanExclusiveShared<<<>>>() 1D block addressing

static const uint MIN_SHORT_ARRAY_SIZE = 4;
static const uint MAX_SHORT_ARRAY_SIZE = 4 * THREADBLOCK_SIZE;
#ifndef NDEBUG
static const uint MAX_BATCH_ELEMENTS = 64 * 1048576;
static const uint MIN_LARGE_ARRAY_SIZE = 4 * THREADBLOCK_SIZE;
static const uint MAX_LARGE_ARRAY_SIZE = 4 * THREADBLOCK_SIZE * THREADBLOCK_SIZE;
#endif

//Internal exclusive scan buffer
//static uint *d_Buf;
//  CUDA_CHECK(cudaMalloc((void **)&d_Buf, (MAX_BATCH_ELEMENTS / (4 * THREADBLOCK_SIZE)) * sizeof(uint)));


static int nextpo2(unsigned int v)
{
     // compute the next highest power of 2 of 32-bit v

    v--;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    v++;

    return v;
}

__global__ void scanExclusiveRidiculous(uint *d_Dst, uint *d_Src, uint arrayLength)
{
    int s = 0;
    for(int i = 0; i < arrayLength; ++i)
    {
	d_Dst[i] = s;
	s += d_Src[i];
    }
}

//block footprint in terms of touched global entries
static const int BFP = 4 * THREADBLOCK_SIZE;

void scanExclusiveShort(cudaStream_t stream, uint *d_Dst, uint *d_Src, uint arrayLength)
{
    //printf("hello short\n");
 
    //Check supported size range
    assert((arrayLength >= MIN_SHORT_ARRAY_SIZE) && (arrayLength <= MAX_SHORT_ARRAY_SIZE));

    //Check total batch size limit
    assert(arrayLength <= MAX_BATCH_ELEMENTS);

    //Check all threadblocks to be fully packed with data
    //assert(arrayLength % (4 * THREADBLOCK_SIZE) == 0);

    scanExclusiveShared<<<(arrayLength + BFP-1) / BFP, THREADBLOCK_SIZE, 0, stream>>>(
        (uint4 *)d_Dst,
        (uint4 *)d_Src,
	nextpo2(arrayLength), arrayLength);

    CUDA_CHECK(cudaPeekAtLastError());
}

void scanExclusiveLarge(cudaStream_t stream, uint *d_Dst, uint *d_Buf, uint *d_Src, uint arrayLength)
{
    //printf("hello large\n");
   
    const int next_size = nextpo2(arrayLength);
    
    //Check supported size range
    assert((arrayLength >= MIN_LARGE_ARRAY_SIZE) && (arrayLength <= MAX_LARGE_ARRAY_SIZE));

    //Check total batch size limit
    assert(arrayLength <= MAX_BATCH_ELEMENTS);

    const int nfineblocks = (arrayLength + BFP - 1) / BFP;
    
    scanExclusiveShared<<< nfineblocks, THREADBLOCK_SIZE, 0, stream>>>(
        (uint4 *)d_Dst,
        (uint4 *)d_Src,
        4 * THREADBLOCK_SIZE,
	arrayLength
    );
    CUDA_CHECK(cudaPeekAtLastError());

    //Not all threadblocks need to be packed with input data:
    //inactive threads of highest threadblock just don't do global reads and writes
    const uint blockCount2 =  (nfineblocks + THREADBLOCK_SIZE - 1) / THREADBLOCK_SIZE; 
    
    scanExclusiveShared2<<< blockCount2, THREADBLOCK_SIZE, 0, stream>>>(
        (uint *)d_Buf,
        (uint *)d_Dst,
        (uint *)d_Src,
        nfineblocks,
        nextpo2(nfineblocks),
	arrayLength
    );
    CUDA_CHECK(cudaPeekAtLastError()); 

    uniformUpdate<<< (arrayLength + BFP - 1) / BFP, THREADBLOCK_SIZE, 0, stream>>>(
        (uint4 *)d_Dst,
        (uint *)d_Buf,
	arrayLength
	);
    CUDA_CHECK(cudaPeekAtLastError());
}

void ScanEngine::exclusive(cudaStream_t stream, uint *d_Dst, uint *d_Src, uint arrayLength)
{	
    if (arrayLength <  MIN_SHORT_ARRAY_SIZE)
    {
	scanExclusiveRidiculous<<<1, 1, 0, stream>>>(d_Dst, d_Src, arrayLength);
    }
    else
    {
	if (arrayLength < MAX_SHORT_ARRAY_SIZE)
	    scanExclusiveShort(stream, d_Dst, d_Src, arrayLength);
	else
	{
	    if (str2buf[stream] == NULL)
	    {
		str2buf[stream] = new SimpleDeviceBuffer<uint>;
	    }
	    
	    str2buf[stream]->resize((arrayLength + BFP - 1) / BFP);
	    printf("BUF HAS %d entries\n", str2buf[stream]->size);
	    scanExclusiveLarge(stream, d_Dst,  str2buf[stream]->data, d_Src, arrayLength);
	}
    }
}

ScanEngine::~ScanEngine()
{
    for(std::map< cudaStream_t, SimpleDeviceBuffer<uint> *>::iterator it = str2buf.begin(); it != str2buf.end(); ++it)
	delete it->second;
}