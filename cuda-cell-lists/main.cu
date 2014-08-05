#include <unistd.h>
#include <cstdio>
#include <cassert>
#include <thrust/device_vector.h>
#include <set>

using namespace thrust;
const int nwarps = 8;
//const int ILP = 4;

__device__ int  blockscount = 0;
 
template<int ILP>
__global__ void acquire_local_offset(const float * const ys,
				     const float * const zs,
				     const int np,
				     const float invrc, const int3 ncells, 
				     const float3 domainstart,
				     int * const yzcid,
				     int * const localoffsets,
				     int * const global_yzhisto,
				     int * const global_yzscan)
{
    extern __shared__ int yzhisto[];

    assert(blockDim.y == 1);
    assert(blockDim.x == warpSize * nwarps);

    const int tid = threadIdx.x;
    const int gsize = gridDim.x * blockDim.x;
    const int nhisto = ncells.y * ncells.z;

    const int tile = blockIdx.x * blockDim.x;
    
    if (tile >= np)
	return;
    
    {
	for(int i = tid ; i < nhisto; i += blockDim.x)
	    yzhisto[i] = 0;
 
	float y[ILP], z[ILP];
	for(int j = 0; j < ILP; ++j)
	{
	    const int g = tile + tid + gsize * j;

	    y[j] = z[j] = -1;

	    if (g < np)
	    { 
		y[j] = ys[g];
		z[j] = zs[g];
	    }
	}

	__syncthreads();
	
	int entries[ILP], offset[ILP];
	for(int j = 0; j < ILP; ++j)
	{
	    const int g = tile + tid + gsize * j;
	    
	    int ycid = (int)(floor(y[j] - domainstart.y) * invrc);
	    int zcid = (int)(floor(z[j] - domainstart.z) * invrc);
	    
	    assert(ycid >= 0 && ycid < ncells.y);
	    assert(zcid >= 0 && zcid < ncells.z);

	    entries[j] = -1;
	    offset[j] = -1;

	    if (g < np)
	    {
		entries[j] =  ycid + ncells.y * zcid;
		offset[j] = (tid % 3   > 0) ? 0 : atomicAdd(yzhisto + entries[j], 1);
	    }
	}

	__syncthreads();
	
	for(int i = tid ; i < nhisto; i += blockDim.x)
	    yzhisto[i] = atomicAdd(global_yzhisto + i, yzhisto[i]);

	__syncthreads();

	for(int j = 0; j < ILP; ++j)
	{
	    const int g = tile + tid + gsize * j;
	    
	    if (g < np)
	    {
		yzcid[g] = entries[j];
		localoffsets[g] = offset[j] + yzhisto[entries[j]];
	    }
	}
    }

    __shared__ bool lastone;

    if (tid == 0)
    {
	lastone = gridDim.x - 1 == atomicAdd(&blockscount, 1);
	
	if (lastone)
	    blockscount = 0;
    }

    __syncthreads();
        
    if (lastone)
    {
	for(int i = tid ; i < nhisto; i += blockDim.x)
	    yzhisto[i] = global_yzhisto[i];

	const int bwork = blockDim.x * ILP;
	for(int tile = 0; tile < nhisto; tile += bwork)
	{
	    const int n = min(bwork, nhisto - tile);

	    __syncthreads();
	    
	    if (tile > 0 && tid == 0)
		yzhisto[tile] += yzhisto[tile - 1];
	    
	    for(int l = 1; l < n; l <<= 1)
	    {
		__syncthreads();
		
		int tmp[ILP];

		for(int j = 0; j < ILP; ++j)
		{
		    const int d = tid + j * blockDim.x;
		    
		    tmp[j] = 0;

		    if (d >= l && d < n) 
			tmp[j] = yzhisto[d + tile] + yzhisto[d + tile - l];
		}

		__syncthreads();

		for(int j = 0; j < ILP; ++j)
		{
		    const int d = tid + j * blockDim.x;

		    if (d >= l && d < n) 
			yzhisto[d + tile] = tmp[j];
		}
	    }
	}

	for(int i = tid ; i < nhisto; i += blockDim.x)
	    global_yzscan[i] = i == 0 ? 0 : yzhisto[i - 1];
	
    }
}

texture<int, cudaTextureType1D> texScanYZ;

template<int ILP>
__global__ void scatter_data(const int * const localoffsets,
			const int * const yzcids,
			const int np,
			int * const outid)
{
    for(int j = 0; j < ILP; ++j)
    {
	const int g = threadIdx.x + blockDim.x * (j + ILP * blockIdx.x);

	if (g < np)
	{
	    const int yzcid = yzcids[g];
	    const int localoffset = localoffsets[g];
	    const int base = tex1Dfetch(texScanYZ, yzcid);
	
	    const int entry = base + localoffset;

	    outid[entry] = g;
	}
    }
}
					

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

void myfill(device_vector<float>& d, const double ext)
{
    const int N = d.size();
    host_vector<float> v(N);
    printf("N is %d\n", N);
    for(int i = 0; i < N; ++i)
	v[i] = -0.5 * ext + drand48() * ext;

    copy(v.begin(), v.end(), d.begin());
}

template<typename T>
T * _ptr(device_vector<T>& v)
{
    return raw_pointer_cast(&v[0]);
}

int main()
{
    const int XL = 40;
    const int YL = 40;
    const int ZL = 40;

    const int N = 3 * XL * YL * ZL;
    const float invrc = 1;
    device_vector<float> xp(N), yp(N), zp(N);
    device_vector<float> xv(N), yv(N), zv(N);

    myfill(xp, XL);
    myfill(yp, YL);
    myfill(zp, ZL);

    myfill(xv, 1);
    myfill(yv, 1);
    myfill(zv, 1);

    //best case scenario
    //if (false)
    {
	for(int i = 0; i < N; ++i)
	{
	    const int cid = i / 3;
	    const int xcid = cid % XL;
	    const int ycid = (cid / XL) % YL;
	    const int zcid = cid / XL / YL;

	    xp[i] = -0.5 * XL + xcid + 0.5 + 0.1 * (drand48() - 0.5);
	    yp[i] = -0.5 * YL + ycid + 0.5 + 0.1 * (drand48() - 0.5);
	    zp[i] = -0.5 * ZL + zcid + 0.5 + 0.1 * (drand48() - 0.5);
	}
    }
    
    printf("my fill is done\n");

    device_vector<int> loffsets(N), yzcid(N);
    device_vector<int> outid(N);
    
    int3 ncells = make_int3((int)XL, (int)YL, (int)ZL);
    float3 domainstart = make_float3(-0.5 * XL, - 0.5 * YL, - 0.5 * ZL);
    

    device_vector<int> yzhisto(ncells.y * ncells.z), dyzscan(ncells.y * ncells.z);

    size_t textureoffset = 0;
    cudaChannelFormatDesc fmt = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindSigned);
    texScanYZ.channelDesc = fmt;
    texScanYZ.filterMode = cudaFilterModePoint;
    texScanYZ.mipmapFilterMode = cudaFilterModePoint;
    texScanYZ.normalized = 0;
   
	
    //fill(yzscan.begin(), yzscan.end(), 0);

    //CUDA_CHECK(cudaMemset(blockscount, 0, sizeof(int)));
    const int blocksize = 32 * nwarps;
    cudaEvent_t evstart, evacquire, evscatter;
    CUDA_CHECK(cudaEventCreate(&evstart));
    CUDA_CHECK(cudaEventCreate(&evacquire));
    CUDA_CHECK(cudaEventCreate(&evscatter));
    CUDA_CHECK(cudaEventRecord(evstart));
    static const int ILP = 4; 
    

    const int nblocks = (N + blocksize * ILP - 1)/ (blocksize * ILP) ;
    acquire_local_offset<ILP><<<nblocks, blocksize, sizeof(int) * ncells.y * ncells.z>>>
	(_ptr(yp), _ptr(zp), N, invrc, ncells, domainstart, _ptr(yzcid),  _ptr(loffsets), _ptr(yzhisto), _ptr(dyzscan));
    CUDA_CHECK(cudaPeekAtLastError());

    {
	cudaThreadSynchronize();
	//sleep(2);
	//exit(0);
    }
    
    CUDA_CHECK(cudaEventRecord(evacquire));

    
     {
	cudaBindTexture(&textureoffset, &texScanYZ, _ptr(dyzscan), &fmt, sizeof(int) * ncells.y * ncells.z);
	
	scatter_data<ILP><<<(N + 256 * ILP - 1) / (256 * ILP), 256>>>(_ptr(loffsets), _ptr(yzcid), N, _ptr(outid));
     }
    
    CUDA_CHECK(cudaEventRecord(evscatter));
    CUDA_CHECK(cudaPeekAtLastError());
     
    cudaThreadSynchronize();

    //sleep(.1);
#ifndef NDEBUG
    {
	host_vector<float> y = yp, z = zp;
	
	host_vector<int> yzhist(ncells.y * ncells.z);
	
	for(int i = 0; i < N; ++i)
	{
	    int ycid = (int)(floor(y[i] - domainstart.y) * invrc);
	    int zcid = (int)(floor(z[i] - domainstart.z) * invrc);

	    const int entry = ycid + ncells.y * zcid;
	    yzhist[entry]++;
	}

	std::set<int> subids[ncells.y * ncells.z];
	
	printf("reading global histo: \n");
	int s = 0;
	for(int i = 0; i < yzhisto.size(); ++i)
	{
	    printf("%d reading %d ref is %d\n", i, (int)yzhisto[i], (int)yzhist[i]);
	    assert(yzhisto[i]  == yzhist[i]);
	    s += yzhisto[i];

	    for(int k = 0; k < yzhist[i]; ++k)
		subids[i].insert(k);
	}
	printf("s == %d is equal to %d == N\n", s , N);
	assert(s == N);
	
	for(int i = 0; i < N; ++i)
	{
	    int ycid = (int)(floor(y[i] - domainstart.y) * invrc);
	    int zcid = (int)(floor(z[i] - domainstart.z) * invrc);

	    const int entry = ycid + ncells.y * zcid;

	    const int loff = loffsets[i];
	    const int en = yzcid[i];

	    assert(en == entry);

	    assert(subids[en].find(loff) != subids[en].end());
	    subids[en].erase(loff);
	}

	for(int i = 0; i < yzhisto.size(); ++i)
	    assert(subids[i].size() == 0);

	printf("first battery verifications passed.\n");
	//assert(false);
    }

    {
	
	int s = 0;
	for(int i = 0; i < dyzscan.size(); ++i)
	{
	    //printf("%d -> %d (%d)\n", i, (int)dyzscan[i], (int) yzhisto[i]);
	  

	    assert(dyzscan[i] == s);

	      s += yzhisto[i];
	}
    }

    {
	host_vector<int> lut = dyzscan;
	
	for(int i = 0; i < N; ++i)
	{
	    const int landing = outid[i];
	    	    
	    const int entry = yzcid[landing];
	    const int base = lut[entry];
	    const int offset = loffsets[landing];

//	    printf("%d: %d -> %d\n", i, landing, base + offset);
	    assert(i == base + offset);
	    
	}
  
	printf("second battery verification passed\n"); 
    }
	
#endif

    CUDA_CHECK(cudaEventSynchronize(evscatter));
    float tacquirems;
    CUDA_CHECK(cudaEventElapsedTime(&tacquirems, evstart, evacquire));
    float tscatterms;
    CUDA_CHECK(cudaEventElapsedTime(&tscatterms, evacquire, evscatter));
    printf("nblocks %d (bs %d) -> %d blocks per sm, active warps per sm %d \n", nblocks, blocksize, nblocks / 7, 3 * nwarps);
    printf("acquiring time... %f ms\n", tacquirems);
    printf("scattering time... %f ms\n", tscatterms);
    printf("one 2read-1write sweep should take about %.3f ms\n", 1e3 * N * 3 * 4/ (90.0 * 1024 * 1024 * 1024)); 
 
    CUDA_CHECK(cudaEventDestroy(evstart));
    CUDA_CHECK(cudaEventDestroy(evacquire));
    
    //  sleep(3);
    printf("test is done\n");
   
    return 0;
}