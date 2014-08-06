#include <unistd.h>
#include <cstdio>
#include <cassert>
#include <thrust/device_vector.h>
#include <set>

using namespace thrust;
const int nwarps = 8*2*2;

__device__ int  blockscount = 0;
 
template<int ILP, int SLOTS>
__global__ void yzhistogram(const float * const ys,
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
    const int slot = tid % (SLOTS);
    const int gsize = gridDim.x * blockDim.x;
    const int nhisto = ncells.y * ncells.z;

    const int tile = blockIdx.x * blockDim.x;
    
    if (tile >= np)
	return;
        
    for(int i = tid ; i < SLOTS * nhisto; i += blockDim.x)
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
	    offset[j] = atomicAdd(yzhisto + entries[j] + slot * nhisto, 1);
	}
    }

    __syncthreads();
	
    for(int s = 1; s < SLOTS; ++s)
    {
	for(int i = tid ; i < nhisto; i += blockDim.x)
	    yzhisto[i + s * nhisto] += yzhisto[i + (s - 1) * nhisto];

	__syncthreads();
    }

    if (slot > 0)
	for(int j = 0; j < ILP; ++j)
	    offset[j] += yzhisto[entries[j] + (slot - 1) * nhisto];
	
    __syncthreads();
	
    for(int i = tid ; i < nhisto; i += blockDim.x)
	yzhisto[i] = atomicAdd(global_yzhisto + i, yzhisto[i + (SLOTS - 1) * nhisto]);

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
texture<float, cudaTextureType1D> texParticles;

template<int YCPB>
__global__ void xgather(const int * const ids, const int np, const float invrc, const int3 ncells, const float3 domainstart,
			int * const starts, int * const counts,
			float * const xyzuvw, const int bufsize)
{
    assert(gridDim.x == 1 && gridDim.y == ncells.y / YCPB && gridDim.z == ncells.z);
    assert(blockDim.x == warpSize);
    assert(blockDim.y == YCPB);
    
    extern __shared__ volatile int allhisto[];
    volatile int * const xhisto = &allhisto[ncells.x * threadIdx.y];
    volatile int * const loffset = &allhisto[YCPB * ncells.x + bufsize * threadIdx.y];
    volatile int * const reordered = &allhisto[YCPB * ncells.x + bufsize * (YCPB + threadIdx.y)];

    const int tid = threadIdx.x;
    const int yzcid = (threadIdx.y + YCPB * blockIdx.y) + ncells.y * blockIdx.z;
    const int start = tex1Dfetch(texScanYZ, yzcid);
    const int count = tex1Dfetch(texCountYZ, yzcid);
    assert(count < bufsize);
    
    for(int i = tid; i < count; i += warpSize)
	xhisto[i] = 0;
 
    for(int i = tid; i < count; i += warpSize)
    {
	const int g = start + i;

 	const int id = ids[g];
	const float x = tex1Dfetch(texParticles, id);
	const int xcid = (int)(invrc * (x - domainstart.x));
	
	const int val = atomicAdd((int *)(xhisto + xcid), 1);
	assert(xcid < ncells.x);
	assert(i < bufsize);
	
	loffset[i] = val |  (xcid << 16);
    }

    //__threadfence_block();
    
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
    for(int i = tid; i < nfloats; i += warpSize)
    {
	const int c = i % 6;
	const int p = reordered[i / 6];
	assert(i / 6 < bufsize);
	
	xyzuvw[6 * start + i] = tex1Dfetch(texParticles, p + np * c);
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
    const int XL = 20;
    const int YL = 20;
    const int ZL = 20;

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
	host_vector<float> x(N), y(N), z(N);
	
	for(int i = 0; i < N; ++i)
	{
	    const int cid = i / 3;
	    const int xcid = cid % XL;
	    const int ycid = (cid / XL) % YL;
	    const int zcid = cid / XL / YL;

	    x [i] = -0.5 * XL + max(0.f, min((float)XL - 0.1, xcid + 0.5 + 2 * (drand48() - 0.5)));
	    y [i] = -0.5 * YL + max(0.f, min((float)YL - 0.1, ycid + 0.5 + 2 * (drand48() - 0.5)));
	    z [i] = -0.5 * ZL + max(0.f, min((float)ZL - 0.1, zcid + 0.5 + 2 * (drand48() - 0.5)));
	}

	xp = x;
	yp = y;
	zp = z;
    }
    
    printf("my fill is done\n");

    device_vector<int> loffsets(N), yzcid(N);
    device_vector<int> outid(N);
    
    int3 ncells = make_int3((int)XL, (int)YL, (int)ZL);
    float3 domainstart = make_float3(-0.5 * XL, - 0.5 * YL, - 0.5 * ZL);
    

    device_vector<int> yzhisto(ncells.y * ncells.z), dyzscan(ncells.y * ncells.z);
    const int ntotcells = ncells.x * ncells.y * ncells.z;
    device_vector<int> start(ntotcells), count(ntotcells);
    device_vector<float> xyzuvw(N * 6), particles_soa(N * 6);

    copy(xp.begin(), xp.end(), particles_soa.begin());
    copy(yp.begin(), yp.end(), particles_soa.begin() + N);
    copy(zp.begin(), zp.end(), particles_soa.begin() + 2 * N);
    copy(xv.begin(), xv.end(), particles_soa.begin() + 3 * N);
    copy(yv.begin(), yv.end(), particles_soa.begin() + 4 * N);
    copy(zv.begin(), zv.end(), particles_soa.begin() + 5 * N);
    

    size_t textureoffset = 0;
    cudaChannelFormatDesc fmt = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindSigned);
    texScanYZ.channelDesc = fmt;
    texScanYZ.filterMode = cudaFilterModePoint;
    texScanYZ.mipmapFilterMode = cudaFilterModePoint;
    texScanYZ.normalized = 0;
    
    texCountYZ.channelDesc = fmt;
    texCountYZ.filterMode = cudaFilterModePoint;
    texCountYZ.mipmapFilterMode = cudaFilterModePoint;
    texCountYZ.normalized = 0;

    texParticles.channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
    texParticles.filterMode = cudaFilterModePoint;
    texParticles.mipmapFilterMode = cudaFilterModePoint;
    texParticles.normalized = 0;
    
    //fill(yzscan.begin(), yzscan.end(), 0);

    //CUDA_CHECK(cudaMemset(blockscount, 0, sizeof(int)));
    const int blocksize = 32 * nwarps;
    cudaEvent_t evstart, evacquire, evscatter, evgather;
    CUDA_CHECK(cudaEventCreate(&evstart));
    CUDA_CHECK(cudaEventCreate(&evacquire));
    CUDA_CHECK(cudaEventCreate(&evscatter));
    CUDA_CHECK(cudaEventCreate(&evgather));
    
    CUDA_CHECK(cudaEventRecord(evstart));
    static const int ILP = 4;
    static const int SLOTS = 3;
    

    const int nblocks = (N + blocksize * ILP - 1)/ (blocksize * ILP) ;
    yzhistogram<ILP, SLOTS><<<nblocks, blocksize, sizeof(int) * ncells.y * ncells.z * SLOTS>>>
	(_ptr(yp), _ptr(zp), N, invrc, ncells, domainstart, _ptr(yzcid),  _ptr(loffsets), _ptr(yzhisto), _ptr(dyzscan));
    //CUDA_CHECK(cudaPeekAtLastError());

    {
	//cudaThreadSynchronize();
	//sleep(2);
	//exit(0);
    }
    
    CUDA_CHECK(cudaEventRecord(evacquire));

    
    {
	cudaBindTexture(&textureoffset, &texScanYZ, _ptr(dyzscan), &fmt, sizeof(int) * ncells.y * ncells.z);
	
	yzscatter<ILP><<<(N + 256 * ILP - 1) / (256 * ILP), 256>>>(_ptr(loffsets), _ptr(yzcid), N, _ptr(outid));
    }
    
    CUDA_CHECK(cudaEventRecord(evscatter));
  
     
    //cudaThreadSynchronize();

    {
	static const int YCPB = 4 ;
	cudaBindTexture(&textureoffset, &texCountYZ, _ptr(yzhisto), &fmt, sizeof(int) * ncells.y * ncells.z);
	cudaBindTexture(&textureoffset, &texParticles, _ptr(particles_soa), &fmt, sizeof(float) * 6 * N);
	const int bufsize = (ncells.x * 3 * 3) / 2;
	xgather<YCPB><<< dim3(1, ncells.y / YCPB, ncells.z), dim3(32, YCPB), sizeof(int) * (ncells.x  + 2 * bufsize) * YCPB>>>(_ptr(outid), N, invrc, ncells, domainstart, _ptr(start), _ptr(count), _ptr(xyzuvw), bufsize);
    }

    CUDA_CHECK(cudaEventRecord(evgather));

    CUDA_CHECK(cudaPeekAtLastError());
    
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
  
	printf("second level   verification passed\n"); 
    }

    {
	host_vector<int> s(start), c(count);
	host_vector<float> aos(xyzuvw);
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

		    //printf("cid %d : my start and count are %d %d\n", cid, mys, myc);
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
	
#endif

    CUDA_CHECK(cudaEventSynchronize(evgather));
    float tacquirems;
    CUDA_CHECK(cudaEventElapsedTime(&tacquirems, evstart, evacquire));
    float tscatterms;
    CUDA_CHECK(cudaEventElapsedTime(&tscatterms, evacquire, evscatter));
    float tgatherms;
    CUDA_CHECK(cudaEventElapsedTime(&tgatherms, evscatter, evgather));
    float ttotalms;
    CUDA_CHECK(cudaEventElapsedTime(&ttotalms, evstart, evgather));
    printf("nblocks %d (bs %d) -> %d blocks per sm, active warps per sm %d \n", nblocks, blocksize, nblocks / 7, 3 * nwarps);
    printf("acquiring time... %f ms\n", tacquirems);
    printf("scattering time... %f ms\n", tscatterms);
    printf("gathering time... %f ms\n", tgatherms);
    printf("total time ... %f ms\n", ttotalms);
    printf("one 2read-1write sweep should take about %.3f ms\n", 1e3 * N * 3 * 4/ (90.0 * 1024 * 1024 * 1024)); 
 
    CUDA_CHECK(cudaEventDestroy(evstart));
    CUDA_CHECK(cudaEventDestroy(evacquire));
    
    //  sleep(3);
    printf("test is done\n");
   
    return 0;
}
