#include <cstdio>
#include <cassert>

struct InfoDPD
{
    int nx, ny, nz, np, nsamples, rsamples_start;
    float XL, YL, ZL;
    float xstart, ystart, zstart, rc, invrc, aij, gamma, sigma, invsqrtdt, sigmaf;
    float * xp, *yp, *zp, *xv, *yv, *zv, *xa, *ya, *za, *rsamples;
    int * starts;
};

__constant__ InfoDPD info;

__global__ void pid2code(int * codes, int * pids)
{
    const int pid = threadIdx.x + blockDim.x * blockIdx.x;

    if (pid >= info.np)
	return;

    const float x = (info.xp[pid] - info.xstart) / info.rc;
    const float y = (info.yp[pid] - info.ystart) / info.rc;
    const float z = (info.zp[pid] - info.zstart) / info.rc;
    
    int ix = (int)floor(x);
    int iy = (int)floor(y);
    int iz = (int)floor(z);
    
    if( !(ix >= 0 && ix < info.nx) ||
	!(iy >= 0 && iy < info.ny) ||
	!(iz >= 0 && iz < info.nz))
	printf("pid %d: oops %f %f %f -> %d %d %d\n", pid, x, y, z, ix, iy, iz);
#if 0 
    assert(ix >= 0 && ix < info.nx);
    assert(iy >= 0 && iy < info.ny);
    assert(iz >= 0 && iz < info.nz);
#else
    ix = max(0, min(info.nx - 1, ix));
    iy = max(0, min(info.ny - 1, iy));
    iz = max(0, min(info.nz - 1, iz));
#endif
    
    codes[pid] = ix + info.nx * (iy + info.nx * iz);
    pids[pid] = pid;
};

__global__ void _gather(const float * input, const int * indices, float * output, const int n)
{
    const int tid = threadIdx.x + blockDim.x * blockIdx.x;

    if (tid < n)
	output[tid] = input[indices[tid]];
}

__device__ void _cellcell(const int s1, const int e1, const int s2, const int e2, const bool self, const int sr,
			  float * const xa, float * const ya, float * const za)
{
    if (threadIdx.x + threadIdx.y == 0)
    {
	for(int c = 0, i = s1; i < e1; ++i)
	    for(int j = self ? i + 1 : s2 ; j < e2; ++j, ++c)
	    {
		float xr = info.xp[i] - info.xp[j];
		float yr = info.yp[i] - info.yp[j];
		float zr = info.zp[i] - info.zp[j];
				
		xr -= info.XL * floorf(0.5f + xr / info.XL);
		yr -= info.YL * floorf(0.5f + yr / info.YL);
		zr -= info.ZL * floorf(0.5f + zr / info.ZL);

		const float rij2 = xr * xr + yr * yr + zr * zr;
		const float invrij = rsqrtf(rij2);
		const float rij = rij2 * invrij;
		const float wr = max((float)0, 1 - rij * info.invrc);
	
		xr *= invrij;
		yr *= invrij;
		zr *= invrij;

		const float rdotv = xr * (info.xv[i] - info.xv[j]) + yr * (info.yv[i] - info.yv[j]) + zr * (info.zv[i] - info.zv[j]);
		const float myrandnr = info.rsamples[(info.rsamples_start + sr + c) % info.nsamples];
#if 0
		assert(myrandnr != -313);
		info.rsamples[(info.rsamples_start + sr + c) % info.nsamples] = -313;
#endif

		const float strength = (info.aij - info.gamma * wr * rdotv + info.sigmaf * myrandnr) * wr;
		const float xf = strength * xr;
		const float yf = strength * yr;
		const float zf = strength * zr;

		assert(!isnan(xf));
		assert(!isnan(yf));
		assert(!isnan(zf));

		xa[i] += xf;
		ya[i] += yf;
		za[i] += zf;
		 
		xa[j] -= xf;
		ya[j] -= yf;
		za[j] -= zf;
	    }
    }
}

__device__ int _cid(int shiftcode)
{
    int3 indx = make_int3(blockIdx.x + (shiftcode & 1),
			  blockIdx.y + ((shiftcode >> 1) & 1),
			  blockIdx.z + ((shiftcode >> 2) & 1));

    indx.x = (indx.x + info.nx) % info.nx;
    indx.y = (indx.y + info.ny) % info.ny;
    indx.z = (indx.z + info.nz) % info.nz;

    return indx.x + info.nx * (indx.y + info.ny * indx.z);
}

__constant__ int edgeslutcount[4] = {8, 3, 2, 1};
__constant__ int edgeslutstart[4] = {0, 8, 11, 13};
__constant__ int edgeslut[14] = {0, 1, 2, 3, 4, 5, 6, 7, 2, 4, 6, 4, 5, 4};

__global__ void _dpd_forces(float * tmp, int * consumed)
{
    const int idbuf = (blockIdx.x & 1) | ((blockIdx.y & 1) << 1) | ((blockIdx.z & 1) << 2);

    float * const xa = tmp + info.np * (idbuf + 8 * 0);
    float * const ya = tmp + info.np * (idbuf + 8 * 1);
    float * const za = tmp + info.np * (idbuf + 8 * 2);
    
    const bool master = threadIdx.x + threadIdx.y == 0;

    __shared__ volatile int offsetrsamples;
    
    for(int i = 0; i < 4; ++i)
    {
	const int cid1 = _cid(i);
	const int s1 = info.starts[cid1];
	const int e1 = info.starts[cid1 + 1];
	
	const int nentries = edgeslutcount[i];
	const int entrystart = edgeslutstart[i];
	
	for(int j = 0; j < nentries; ++j)
	{
	    const int cid2 = _cid(edgeslut[j + entrystart]);
	    const int s2 = info.starts[cid2];
	    const int e2 = info.starts[cid2 + 1];

	    const bool self = cid1 == cid2;
	    const int rconsumption = self ? (e1 - s1) * (e1 - s1 - 1) / 2 : (e1 - s1) * (e2 - s2); 

	    if(master)
		offsetrsamples = atomicAdd(consumed, rconsumption);

	    __syncthreads();

	    if (offsetrsamples + rconsumption >= info.nsamples)
		return;
	    
	    _cellcell(s1, e1, s2, e2, cid1 == cid2, offsetrsamples, xa, ya, za);
	}
    }
}

__global__ void _reduce(float * tmp)
{
    const int pid = threadIdx.x + blockDim.x * blockIdx.x;

    if (pid < info.np)
    {
	float xa = 0;
	for(int idbuf = 0; idbuf < 8; ++idbuf)
	    xa += tmp[pid + info.np * (idbuf + 8 * 0)];

	float ya = 0;
	for(int idbuf = 0; idbuf < 8; ++idbuf)
	    ya += tmp[pid + info.np * (idbuf + 8 * 1)];
	
	float za = 0;	
    	for(int idbuf = 0; idbuf < 8; ++idbuf)
	    za += tmp[pid + info.np * (idbuf + 8 * 2)];

	info.xa[pid] = xa;
	info.ya[pid] = ya;
	info.za[pid] = za;
    }
}

#include <unistd.h>

#include <curand.h>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/binary_search.h>

using namespace thrust;

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

template<typename T> T * _ptr(device_vector<T>& v) { return raw_pointer_cast(v.data()); }

void _reorder(device_vector<float>& v, device_vector<int>& indx)
{
    device_vector<float> tmp(v.begin(), v.end());
   
    _gather<<<(v.size() + 127) / 128, 128>>>(_ptr(tmp), _ptr(indx), _ptr(v), v.size());

    CUDA_CHECK(cudaPeekAtLastError());
    CUDA_CHECK(cudaThreadSynchronize());
}

class RRingBuffer
{
    const int n;
    int s, c, olds;
    float * drsamples;
    curandGenerator_t prng;

protected:

    void _refill(int s, int e)
	{
	    assert(e > s && e <= n);
	    
	    const int multiple = 2;

	    s = s - (s % multiple);
	    e = e + (multiple - (e % multiple));
	    e = min(e, n);
	    
	    curandStatus_t res;
	    res = curandGenerateNormal(prng, drsamples + s, e - s, 0, 1);
	    assert(res == CURAND_STATUS_SUCCESS);
	}
    
public:

    RRingBuffer(const int n): n(n), s(0), olds(0), c(0)
	{
	    curandStatus_t res;
	    res = curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_DEFAULT);
	    //we could try CURAND_RNG_PSEUDO_MTGP32 or CURAND_RNG_PSEUDO_MT19937
	    
	    assert(res == CURAND_STATUS_SUCCESS);
	    res = curandSetPseudoRandomGeneratorSeed(prng, 1234ULL);
	    assert(res == CURAND_STATUS_SUCCESS);
	    
	    CUDA_CHECK(cudaMalloc(&drsamples, sizeof(float) * n));

	    update(n);
	    assert(s == 0);
	}

    ~RRingBuffer()
	{
	    CUDA_CHECK(cudaFree(drsamples));
	    curandStatus_t res = curandDestroyGenerator(prng);
	    assert(res == CURAND_STATUS_SUCCESS);
	}
    
    void update(const int consumed)
	{
	    assert(consumed >= 0 && consumed <= n);

	    c += consumed;
	    assert(c >= 0 && c <= n);
	    
	    if (c > 0.45 * n)
	    {
		const int c1 = min(olds + c, n) - olds;
	    
		if (c1 > 0)
		    _refill(olds, olds + c1);

		const int c2 = c - c1;

		if (c2 > 0)
		    _refill(0, c2);
	    
		olds = (olds + c) % n;
		s = olds;
		c = 0;
	    }
	    else
		s = (olds + c) % n;
	}

    int start() const { return s; }
    float * buffer() const { return drsamples; }
    int nsamples() const { return n; }
};

void forces_dpd_cuda(float * const _xp, float * const _yp, float * const _zp,
		     float * const _xv, float * const _yv, float * const _zv,
		     float * const _xa, float * const _ya, float * const _za,
		     int * const order, const int np,
		     const float rc,
		     const float XL, const float YL, const float ZL,
		     const float aij,
		     const float gamma,
		     const float sigma,
		     const float invsqrtdt,
		     float * const _rsamples, int nsamples)
{
    static bool initialized = false;

    if (!initialized)
    {
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 0);
	if (!prop.canMapHostMemory)
	{
	    printf("Capability zero-copy not there! Aborting now.\n");
	    abort();
	}
	else
	    cudaSetDeviceFlags(cudaDeviceMapHost);

	initialized = true;
    }

    static RRingBuffer * rrbuf = NULL;

    if (rrbuf == NULL)
	rrbuf = new RRingBuffer(50 * np);
     
   
	
    int nx = (int)ceil(XL / rc);
    int ny = (int)ceil(YL / rc);
    int nz = (int)ceil(ZL / rc);
    const int ncells = nx * ny * nz;
    
    device_vector<int> starts(ncells + 1);
    
    device_vector<float> xp(_xp, _xp + np), yp(_yp, _yp + np), zp(_zp, _zp + np),
	xv(_xv, _xv + np), yv(_yv, _yv + np), zv(_zv, _zv + np);	

    device_vector<float> xa(np), ya(np), za(np);
    fill(xa.begin(), xa.end(), 0);
    fill(ya.begin(), ya.end(), 0);
    fill(za.begin(), za.end(), 0);
    
    InfoDPD c;
    c.nx = nx;
    c.ny = ny;
    c.nz = nz;
    c.np = np;
    c.XL = XL;
    c.YL = YL;
    c.ZL = ZL;
    c.xstart = -XL * 0.5; 
    c.ystart = -YL * 0.5; 
    c.zstart = -ZL * 0.5; 
    c.rc = rc;
    c.invrc = 1.f / rc;
    c.aij = aij;
    c.gamma = gamma;
    c.sigma = sigma;
    c.invsqrtdt = invsqrtdt;
    c.sigmaf = sigma * invsqrtdt;
    c.xp = _ptr(xp);
    c.yp = _ptr(yp);
    c.zp = _ptr(zp);
    c.xv = _ptr(xv);
    c.yv = _ptr(yv);
    c.zv = _ptr(zv);
    c.xa = _ptr(xa);
    c.ya = _ptr(ya);
    c.za = _ptr(za);
    c.nsamples = rrbuf->nsamples();
    c.rsamples = rrbuf->buffer();
    c.rsamples_start = rrbuf->start();

    device_vector<float> rsamples;
    if (_rsamples != NULL)
    {
	rsamples.resize(nsamples);
	copy(_rsamples, _rsamples + nsamples, rsamples.begin());

	c.nsamples = nsamples;
	c.rsamples = _ptr(rsamples);
	c.rsamples_start = 0;
    }
    else
	nsamples = rrbuf->nsamples();
    
    c.starts = _ptr(starts);
    
    CUDA_CHECK(cudaMemcpyToSymbol(info, &c, sizeof(c)));

    device_vector<int> codes(np), pids(np);
    pid2code<<<(np + 127) / 128, 128>>>(_ptr(codes), _ptr(pids));

    sort_by_key(codes.begin(), codes.end(), pids.begin());

    _reorder(xp, pids);
    _reorder(yp, pids);
    _reorder(zp, pids);
    
    _reorder(xv, pids);
    _reorder(yv, pids);
    _reorder(zv, pids);
    
    device_vector<int> cids(ncells + 1);
    sequence(cids.begin(), cids.end());

    lower_bound(codes.begin(), codes.end(), cids.begin(), cids.end(), starts.begin());

    int * consumed = NULL;
    cudaHostAlloc((void **)&consumed, sizeof(int), cudaHostAllocMapped);
    assert(consumed != NULL);
    *consumed = 0;
    
    {
	float * tmp;

	CUDA_CHECK(cudaMalloc(&tmp, sizeof(float) * np * 24));
	CUDA_CHECK(cudaMemset(tmp, 0, sizeof(float) * np * 24));
	
	int * dconsumed = NULL;
	cudaHostGetDevicePointer(&dconsumed, consumed, 0);
    
	_dpd_forces<<<dim3(c.nx, c.ny, c.nz), dim3(1, 1, 1)>>>(tmp, dconsumed);
	_reduce<<<(np + 127) / 128, 128>>>(tmp);
	
	CUDA_CHECK(cudaPeekAtLastError());
	CUDA_CHECK(cudaThreadSynchronize());
	CUDA_CHECK(cudaFree(tmp));
	
	if (*consumed >= nsamples)
	{
	    printf("done with code %d: consumed: %d\n", 7, *consumed);
	    printf("not a nice situation.\n");
	    abort();
	}

	//printf("consumed: %d\n", *consumed);
    }

    if (_rsamples == NULL)
	rrbuf->update(*consumed);
    
    cudaFreeHost(consumed);
    
    copy(xp.begin(), xp.end(), _xp);
    copy(yp.begin(), yp.end(), _yp);
    copy(zp.begin(), zp.end(), _zp);
	
    copy(xv.begin(), xv.end(), _xv);
    copy(yv.begin(), yv.end(), _yv);
    copy(zv.begin(), zv.end(), _zv);

    copy(xa.begin(), xa.end(), _xa);
    copy(ya.begin(), ya.end(), _ya);
    copy(za.begin(), za.end(), _za);

    if (order != NULL)
	copy(pids.begin(), pids.end(), order);
}