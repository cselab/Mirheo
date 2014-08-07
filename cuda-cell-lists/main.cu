#include <cstdio>
#include <cassert>

texture<float, cudaTextureType1D> texParticles;

__device__ int  blockscount = 0;
 
template<int ILP, int SLOTS, int WARPS>
__global__ void yzhistogram(const int np,
			    const float invrc, const int3 ncells, 
			    const float3 domainstart,
			    int * const yzcid,
			    int * const localoffsets,
			    int * const global_yzhisto,
			    int * const global_yzscan,
			    int * const max_yzcount)
{
    extern __shared__ int shmemhisto[];

    assert(blockDim.y == 1);
    assert(blockDim.x == warpSize * WARPS);

    const int tid = threadIdx.x;
    const int slot = tid % (SLOTS);
    const int gsize = gridDim.x * blockDim.x;
    const int nhisto = ncells.y * ncells.z;

    const int tile = blockIdx.x * blockDim.x;
    
    if (tile >= np)
	return;
        
    for(int i = tid ; i < SLOTS * nhisto; i += blockDim.x)
	shmemhisto[i] = 0;
 
    float y[ILP], z[ILP];
    for(int j = 0; j < ILP; ++j)
    {
	const int g = tile + tid + gsize * j;

	y[j] = z[j] = -1;

	if (g < np)
	{
	    y[j] = tex1Dfetch(texParticles, 1 + 6 * g); 
	    z[j] = tex1Dfetch(texParticles, 2 + 6 * g); 
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
	    offset[j] = atomicAdd(shmemhisto + entries[j] + slot * nhisto, 1);
	}
    }

    __syncthreads();
	
    for(int s = 1; s < SLOTS; ++s)
    {
	for(int i = tid ; i < nhisto; i += blockDim.x)
	    shmemhisto[i + s * nhisto] += shmemhisto[i + (s - 1) * nhisto];

	__syncthreads();
    }

    if (slot > 0)
	for(int j = 0; j < ILP; ++j)
	    offset[j] += shmemhisto[entries[j] + (slot - 1) * nhisto];
	
    __syncthreads();
	
    for(int i = tid ; i < nhisto; i += blockDim.x)
	shmemhisto[i] = atomicAdd(global_yzhisto + i, shmemhisto[i + (SLOTS - 1) * nhisto]);

    __syncthreads();

    for(int j = 0; j < ILP; ++j)
    {
	const int g = tile + tid + gsize * j;
	    
	if (g < np)
	{
	    yzcid[g] = entries[j];
	    localoffsets[g] = offset[j] + shmemhisto[entries[j]];
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
	    shmemhisto[i] = global_yzhisto[i];

	if (max_yzcount != NULL)
	{
	    __syncthreads();

	    int mymax = 0;
	    for(int i = tid ; i < nhisto; i += blockDim.x)
		mymax = max(shmemhisto[i], mymax);

	    for(int L = 16; L > 0; L >>=1)
		mymax = max(__shfl_xor(mymax, L), mymax);

	    __shared__ int maxies[WARPS];
	
	    if (tid % warpSize == 0)
		maxies[tid / warpSize] = mymax;
	
	    __syncthreads();

	    mymax = 0;
	
	    if (tid < WARPS)
		mymax = maxies[tid];

	    for(int L = 16; L > 0; L >>=1)
		mymax = max(__shfl_xor(mymax, L), mymax);

	    if (tid == 0)
		*max_yzcount = mymax;
	}
	
	const int bwork = blockDim.x * ILP;
	for(int tile = 0; tile < nhisto; tile += bwork)
	{
	    const int n = min(bwork, nhisto - tile);

	    __syncthreads();
	    
	    if (tile > 0 && tid == 0)
		shmemhisto[tile] += shmemhisto[tile - 1];
	    
	    for(int l = 1; l < n; l <<= 1)
	    {
		__syncthreads();
		
		int tmp[ILP];

		for(int j = 0; j < ILP; ++j)
		{
		    const int d = tid + j * blockDim.x;
		    
		    tmp[j] = 0;

		    if (d >= l && d < n) 
			tmp[j] = shmemhisto[d + tile] + shmemhisto[d + tile - l];
		}

		__syncthreads();

		for(int j = 0; j < ILP; ++j)
		{
		    const int d = tid + j * blockDim.x;

		    if (d >= l && d < n) 
			shmemhisto[d + tile] = tmp[j];
		}
	    }
	}

	for(int i = tid ; i < nhisto; i += blockDim.x)
	    global_yzscan[i] = i == 0 ? 0 : shmemhisto[i - 1];
    }
}

texture<int, cudaTextureType1D> texScanYZ;

template<int ILP>
__global__ void yzscatter(const int * const localoffsets,
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

texture<int, cudaTextureType1D> texCountYZ;

template<int YCPB>
__global__ void xgather(const int * const ids, const int np, const float invrc, const int3 ncells, const float3 domainstart,
			int * const starts, int * const counts,
			float * const xyzuvw, const int bufsize, int * const order)
{
    assert(gridDim.x == 1 && gridDim.y == ncells.y / YCPB && gridDim.z == ncells.z);
    assert(blockDim.x == warpSize);
    assert(blockDim.y == YCPB);
    
    extern __shared__ volatile int allhisto[];
    volatile int * const xhisto = &allhisto[ncells.x * threadIdx.y];
    volatile int * const loffset = &allhisto[YCPB * ncells.x + bufsize * threadIdx.y];
    volatile int * const reordered = &allhisto[YCPB * ncells.x + bufsize * (YCPB + threadIdx.y)];

    const int tid = threadIdx.x;
    const int ycid = threadIdx.y + YCPB * blockIdx.y;

    if (ycid >= ncells.y)
	return;
    
    const int yzcid = ycid + ncells.y * blockIdx.z;
    const int start = tex1Dfetch(texScanYZ, yzcid);
    const int count = tex1Dfetch(texCountYZ, yzcid);

    if (count >= bufsize)
	return; //something went wrong 
    
    for(int i = tid; i < count; i += warpSize)
	xhisto[i] = 0;
 
    for(int i = tid; i < count; i += warpSize)
    {
	const int g = start + i;

 	const int id = ids[g];

	const float x = tex1Dfetch(texParticles, 6 * id);

	const int xcid = (int)(invrc * (x - domainstart.x));
	
	const int val = atomicAdd((int *)(xhisto + xcid), 1);
	assert(xcid < ncells.x);
	assert(i < bufsize);
	
	loffset[i] = val |  (xcid << 16);
    }
    
    for(int i = tid; i < ncells.x; i += warpSize)
	counts[i + ncells.x * yzcid] = xhisto[i];

    for(int base = 0; base < ncells.x; base += warpSize)
    {
	const int n = min(warpSize, ncells.x - base);
	const int g = base + tid;
	
	int val = (tid == 0 && base > 0) ? xhisto[g - 1] : 0;

	if (tid < n)
	    val += xhisto[g];

	for(int l = 1; l < n; l <<= 1)
	    val += (tid >= l) * __shfl_up(val, l);

	if (tid < n)
	    xhisto[g] = val;
    }

    for(int i = tid; i < ncells.x; i += warpSize)
	starts[i + ncells.x * yzcid] = start + (i == 0 ? 0 : xhisto[i - 1]);
 
    for(int i = tid; i < count; i += warpSize)
    {
	const int entry = loffset[i];
	const int xcid = entry >> 16;
	assert(xcid < ncells.x);
	const int loff = entry & 0xffff;

	const int dest = (xcid == 0 ? 0 : xhisto[xcid - 1]) + loff;

	reordered[dest] = ids[start + i];
    }

    const int nfloats = count * 6;
    const int base = 6 * start;
    
    const int mystart = (32 - (base & 0x1f) + tid) & 0x1f;
    for(int i = mystart; i < nfloats; i += warpSize)
    {
	const int c = i % 6;
	const int p = reordered[i / 6];
	assert(i / 6 < bufsize);
	
	xyzuvw[base + i] = tex1Dfetch(texParticles, c + 6 * p);
    }

    if (order != NULL)
	for(int i = tid; i < count; i += warpSize)
	    order[start + i] = reordered[i];
}

#include <unistd.h>

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

#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/iterator/counting_iterator.h>

using namespace thrust;

template<typename T>
T * _ptr(device_vector<T>& v)
{
    return raw_pointer_cast(&v[0]);
}

#include <utility>

struct FailureTest
{
    int bufsize;
    int * maxstripe, * dmaxstripe;

    FailureTest()
	{
	    cudaDeviceProp prop;
	    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
	    if (!prop.canMapHostMemory)
	    {
		printf("Capability zero-copy not there! Aborting now.\n");
		abort();
	    }
	    else
		CUDA_CHECK(cudaSetDeviceFlags(cudaDeviceMapHost));
    
	    CUDA_CHECK(cudaHostAlloc((void **)&maxstripe, sizeof(int), cudaHostAllocMapped));
	    assert(maxstripe != NULL);
	    CUDA_CHECK(cudaHostGetDevicePointer(&dmaxstripe, maxstripe, 0));
	}

    static void callback_crash(cudaStream_t stream, cudaError_t status, void*  userData )
	{
	    FailureTest& f = *(FailureTest *)userData;
	    
	    if (*f.maxstripe > f.bufsize)
	    {
		printf("Ouch .. I would need to rollback. Maxstripe: %d, bufsize: %d\n", *f.maxstripe, f.bufsize);
		printf("Too late to recover this. Aborting now.\n");
		abort();
	    }
	}
} failuretest;

device_vector<int> loffsets, yzcid, outid, yzhisto, dyzscan;

struct is_gzero
{
  __host__ __device__
  bool operator()(const int &x)
  {
    return  x > 0;
  }
};

void build_clists(float * const device_xyzuvw, int np, const float rc, 
		  const int xcells, const int ycells, const int zcells,
		  const float xdomainstart, const float ydomainstart, const float zdomainstart,
		  int * const host_order, int * device_cellsstart, int * device_cellscount,
		  std::pair<int, int *> * nonemptycells = NULL)
{
    const bool robust = true;
    const float invrc = 1 / rc;
    const float3 domainstart = make_float3(xdomainstart, ydomainstart, zdomainstart);
    const int3 ncells = make_int3(xcells, ycells, zcells);
    
    const float densitynumber = np / (float)(ncells.x * ncells.y * ncells.z);
    const int xbufsize = max(32, (int)(ncells.x * densitynumber * 2));
    
    device_vector<float> xyzuvw_copy(device_ptr<float>(device_xyzuvw), device_ptr<float>(device_xyzuvw + 6 * np));

    device_vector<int> order;
    int * device_order = NULL;
    
    if (host_order != NULL)
    {
	order.resize(np);
	device_order = _ptr(order);
    }
    
    cudaEvent_t evstart, evacquire, evscatter, evgather;
    CUDA_CHECK(cudaEventCreate(&evstart));
    CUDA_CHECK(cudaEventCreate(&evacquire));
    CUDA_CHECK(cudaEventCreate(&evscatter));
    CUDA_CHECK(cudaEventCreate(&evgather));
  
    loffsets.resize(np);
    yzcid.resize(np);
    outid.resize(np);
    yzhisto.resize(ncells.y * ncells.z);
    dyzscan.resize(ncells.y * ncells.z);

    texScanYZ.channelDesc = cudaCreateChannelDesc<int>();
    texScanYZ.filterMode = cudaFilterModePoint;
    texScanYZ.mipmapFilterMode = cudaFilterModePoint;
    texScanYZ.normalized = 0;
    
    texCountYZ.channelDesc = cudaCreateChannelDesc<int>();
    texCountYZ.filterMode = cudaFilterModePoint;
    texCountYZ.mipmapFilterMode = cudaFilterModePoint;
    texCountYZ.normalized = 0;

    texParticles.channelDesc = cudaCreateChannelDesc<float>();
    texParticles.filterMode = cudaFilterModePoint;
    texParticles.mipmapFilterMode = cudaFilterModePoint;
    texParticles.normalized = 0;

    size_t textureoffset = 0;
    CUDA_CHECK(cudaBindTexture(&textureoffset, &texParticles, _ptr(xyzuvw_copy), &texParticles.channelDesc, sizeof(float) * 6 * np));
       
    CUDA_CHECK(cudaEventRecord(evstart));

    {
	static const int ILP = 4;
	static const int SLOTS = 3;
	static const int WARPS = 32;
	const int blocksize = 32 * WARPS;
	const int nblocks = (np + blocksize * ILP - 1)/ (blocksize * ILP);
	//printf("nblocks %d (bs %d) -> %d blocks per sm, active warps per sm %d \n", nblocks, blocksize, nblocks / 7, 3 * WARPS);
	
	yzhistogram<ILP, SLOTS, WARPS><<<nblocks, blocksize, sizeof(int) * ncells.y * ncells.z * SLOTS>>>
	    (np, invrc, ncells, domainstart, _ptr(yzcid),  _ptr(loffsets), _ptr(yzhisto), _ptr(dyzscan), failuretest.dmaxstripe);
    }
    
    CUDA_CHECK(cudaEventRecord(evacquire));
    
    {
	static const int ILP = 4;
	
	CUDA_CHECK(cudaBindTexture(&textureoffset, &texScanYZ, _ptr(dyzscan), &texScanYZ.channelDesc, sizeof(int) * ncells.y * ncells.z));
	
	yzscatter<ILP><<<(np + 256 * ILP - 1) / (256 * ILP), 256>>>(_ptr(loffsets), _ptr(yzcid), np, _ptr(outid));
    }
    
    CUDA_CHECK(cudaEventRecord(evscatter));
    
    {
	static const int YCPB = 2 ;
	CUDA_CHECK(cudaBindTexture(&textureoffset, &texCountYZ, _ptr(yzhisto), &texCountYZ.channelDesc, sizeof(int) * ncells.y * ncells.z));
	
	xgather<YCPB><<< dim3(1, ncells.y / YCPB, ncells.z), dim3(32, YCPB), sizeof(int) * (ncells.x  + 2 * xbufsize) * YCPB>>>
	    (_ptr(outid), np, invrc, ncells, domainstart, device_cellsstart, device_cellscount, device_xyzuvw, xbufsize, device_order);
    }
     
    if (!robust)
    {
	failuretest.bufsize = xbufsize;
	CUDA_CHECK(cudaStreamAddCallback(0, failuretest.callback_crash, &failuretest, 0));
    }
    else
    {
	CUDA_CHECK(cudaEventSynchronize(evacquire));

	if (*failuretest.maxstripe > xbufsize)
	{
	    printf("Ooops: maxstripe %d > bufsize %d.\nRecovering now...\n", *failuretest.maxstripe, xbufsize);
	    
	    const int xbufsize = *failuretest.maxstripe;
	    
	    xgather<1><<< dim3(1, ncells.y, ncells.z), dim3(32), sizeof(int) * (ncells.x  + 2 * xbufsize)>>>
		(_ptr(outid), np, invrc, ncells, domainstart, device_cellsstart, device_cellscount, device_xyzuvw, xbufsize, device_order);

	    cudaError_t status = cudaPeekAtLastError();

	    if (status != cudaSuccess)
	    {
		printf("Could not roll back. Aborting now.\n");
		abort();
	    }
	    else
		printf("Recovery succeeded.\n");
	}
    }

    CUDA_CHECK(cudaEventRecord(evgather));
    
    CUDA_CHECK(cudaEventSynchronize(evgather));
   
    CUDA_CHECK(cudaPeekAtLastError());
    float tacquirems;
    CUDA_CHECK(cudaEventElapsedTime(&tacquirems, evstart, evacquire));
    float tscatterms;
    CUDA_CHECK(cudaEventElapsedTime(&tscatterms, evacquire, evscatter));
    float tgatherms;
    CUDA_CHECK(cudaEventElapsedTime(&tgatherms, evscatter, evgather));
    float ttotalms;
    CUDA_CHECK(cudaEventElapsedTime(&ttotalms, evstart, evgather));
    
    printf("acquiring time... %f ms\n", tacquirems);
    printf("scattering time... %f ms\n", tscatterms);
    printf("gathering time... %f ms\n", tgatherms);
    printf("total time ... %f ms\n", ttotalms);
    printf("one 2read-1write sweep should take about %.3f ms\n", 1e3 * np * 3 * 4/ (90.0 * 1024 * 1024 * 1024));
    printf("maxstripe was %d and bufsize is %d\n", *failuretest.maxstripe, xbufsize);
 
    CUDA_CHECK(cudaEventDestroy(evstart));
    CUDA_CHECK(cudaEventDestroy(evacquire));
    CUDA_CHECK(cudaEventDestroy(evscatter));
    CUDA_CHECK(cudaEventDestroy(evgather));

    if (nonemptycells != NULL)
    {
	assert(nonemptycells->second != NULL);

	const int ntotcells = ncells.x * ncells.y * ncells.z;
	const int nonempties = copy_if(counting_iterator<int>(0), counting_iterator<int>(ntotcells), 
				       device_ptr<int>(device_cellscount), device_ptr<int>(nonemptycells->second), is_gzero())
	    - device_ptr<int>(nonemptycells->second);
	
	nonemptycells->first = nonempties;
    }

    if (host_order != NULL)
	copy(order.begin(), order.end(), host_order);
}

#include <set>

void myfill(device_vector<float>& d, const double ext)
{
    const int N = d.size();
    host_vector<float> v(N);
    printf("N is %d\n", N);
    for(int i = 0; i < N; ++i)
	v[i] = -0.5 * ext + drand48() * ext;

    copy(v.begin(), v.end(), d.begin());
}

int main()
{
    const int XL = 40;
    const int YL = 40;
    const int ZL = 40;

    const float densitynumber = 3;
    const int N = densitynumber * XL * YL * ZL;
    const float rc = 1;
    const float invrc = 1 / rc;
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
	host_vector<float> x(N), y(N), z(N);
	
	for(int i = 0; i < N; ++i)
	{
	    const int cid = i / 3;
	    const int xcid = cid % XL;
	    const int ycid = (cid / XL) % YL;
	    const int zcid = cid / XL / YL;

	    x [i] = -0.5 * XL + max(0.f, min((float)XL - 0.1, xcid + 0.5 + 3 * (drand48() - 0.5)));
	    y [i] = -0.5 * YL + max(0.f, min((float)YL - 0.1, ycid + 0.5 + 3 * (drand48() - 0.5)));
	    z [i] = -0.5 * ZL + max(0.f, min((float)ZL - 0.1, zcid + 0.5 + 3 * (drand48() - 0.5)));
	}

	xp = x;
	yp = y;
	zp = z;
    }
    
    printf("my fill is done\n");
    
    int3 ncells = make_int3((int)ceil(XL / rc), (int)ceil(YL / rc), (int)ceil(ZL/rc));
    float3 domainstart = make_float3(-0.5 * XL, - 0.5 * YL, - 0.5 * ZL);
    
    const int ntotcells = ncells.x * ncells.y * ncells.z;
    device_vector<int> start(ntotcells), count(ntotcells);
    device_vector<float> particles_aos(N * 6);
    {
	host_vector<float> _xp(xp), _yp(yp), _zp(zp), _xv(xv), _yv(yv), _zv(zv), _particles_aos(6 * N);
	
	for(int i = 0; i < N; ++i)
	{
	    _particles_aos[0 + 6 * i] = _xp[i];
	    _particles_aos[1 + 6 * i] = _yp[i];
	    _particles_aos[2 + 6 * i] = _zp[i];
	    _particles_aos[3 + 6 * i] = _xv[i];
	    _particles_aos[4 + 6 * i] = _yv[i];
	    _particles_aos[5 + 6 * i] = _zv[i];
	}
	
	particles_aos = _particles_aos;
    }

    host_vector<int> order(N);
    
    build_clists(_ptr(particles_aos), N,  rc, 
		 ncells.x, ncells.y, ncells.z,
		 -0.5 * XL, -0.5 * YL, -0.5 * ZL,
		 &order.front(), _ptr(start), _ptr(count), NULL);
    
#ifndef NDEBUG
    CUDA_CHECK(cudaThreadSynchronize());
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
	
	//printf("reading global histo: \n");
	int s = 0;
	for(int i = 0; i < yzhisto.size(); ++i)
	{
	    // printf("%d reading %d ref is %d\n", i, (int)yzhisto[i], (int)yzhist[i]);
	    assert(yzhisto[i]  == yzhist[i]);
	    s += yzhisto[i];

	    for(int k = 0; k < yzhist[i]; ++k)
		subids[i].insert(k);
	}
	//printf("s == %d is equal to %d == N\n", s , N);
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

	printf("first level   verifications passed.\n");

	const int mymax = *max_element(yzhist.begin(), yzhist.end());
	printf("mymax: %d maxstripe: %d\n", mymax, *failuretest.maxstripe);
	assert(mymax == *failuretest.maxstripe);
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
	    //printf("%d: %d -> %d\n", i, landing, base + offset);
	    assert(i == base + offset);
	}
  
	printf("second level   verification passed\n"); 
    }

    {
	host_vector<int> s(start), c(count);
	host_vector<float> aos(particles_aos);
	host_vector<bool> marked(N);

	//printf("start[0] : %d\n", (int)start[0]);
	assert(start[0] == 0);
	
	for(int iz = 0; iz < ZL; ++iz)
	    for(int iy = 0; iy < YL; ++iy)
		for(int ix = 0; ix < XL; ++ix)
		{
		    const int cid = ix + XL * (iy + YL * iz);
		    const int mys = s[cid];
		    const int myc = c[cid];

		    //intf("cid %d : my start and count are %d %d\n", cid, mys, myc);
		    assert(mys >= 0 && mys < N);
		    assert(myc >= 0 && myc <= N);

		    for(int i = mys; i < mys + myc; ++i)
		    {
			assert(!marked[i]);
			const float x = aos[0 + 6 * i];
			const float y = aos[1 + 6 * i];
			const float z = aos[2 + 6 * i];
			
			const float xcheck = x - domainstart.x;
			const float ycheck = y - domainstart.y;
			const float zcheck = z - domainstart.z;

			//printf("checking p %d: %f %f %f  ref: %d %d %d\n", i, xcheck , ycheck, zcheck, ix, iy, iz);
			assert(xcheck >= ix && xcheck < ix + 1);
			assert(ycheck >= iy && ycheck < iy + 1);
			assert(zcheck >= iz && zcheck < iz + 1);
						
			marked[i] = true;
		    }
		}

	printf("third-level verification passed.\n");
    }

    {
	std::set<int> ids;
	for(int i = 0; i < N; ++i)
	{
	    const int key = order[i];
	    //printf("%d : %d\n", i, key);
	    assert(key >= 0);
	    assert(key < N);
	    assert(ids.find(key) == ids.end());
	    ids.insert(key);
	}
	
    }

     printf("test is done\n");
#endif

    return 0;
}

