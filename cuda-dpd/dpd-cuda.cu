#include <cstdio>
#include <cassert>

struct InfoDPD
{
    int nx, ny, nz, np, nsamples;
    float XL, YL, ZL;
    float xstart, ystart, zstart, rc, aij, gamma, sigma, invsqrtdt;
    float * xp, *yp, *zp, *xv, *yv, *zv, *xa, *ya, *za, *rsamples;
    int * starts;
};

__constant__ InfoDPD info;

__global__ void pid2code(int * codes, int * pids)
{
    const int pid = threadIdx.x + blockDim.x * blockIdx.x;

    if (pid >= info.np)
	return;

    const int ix = (int)((info.xp[pid] - info.xstart) / info.rc);
    const int iy = (int)((info.yp[pid] - info.ystart) / info.rc);
    const int iz = (int)((info.zp[pid] - info.zstart) / info.rc);
    
    assert(ix >= 0 && ix < info.nx);
    assert(iy >= 0 && iy < info.ny);
    assert(iz >= 0 && iz < info.nz);
    
    codes[pid] = ix + info.nx * (iy + info.nx * iz);
    pids[pid] = pid;
};

__global__ void _gather(const float * input, const int * indices, float * output, const int n)
{
    const int tid = threadIdx.x + blockDim.x * blockIdx.x;

    if (tid < n)
	output[tid] = input[indices[tid]];
}

__device__ void _ccfc(const int s1, const int e1, const int s2, const int e2, const bool self, const int sr)
{
    if (threadIdx.x + threadIdx.y == 0)
    {
	for(int c = 0, i = s1; i < e1; ++i)
	    for(int j = self ? i + 1 : s2 ; j < e2; ++j, ++c)
	    {
		float xr = info.xp[i] - info.xp[j];
		float yr = info.yp[i] - info.yp[j];
		float zr = info.zp[i] - info.zp[j];
				
		xr -= info.XL * floor(0.5f + xr / info.XL);
		yr -= info.YL * floor(0.5f + yr / info.YL);
		zr -= info.ZL * floor(0.5f + zr / info.ZL);

		float rij = sqrtf(xr * xr + yr * yr + zr * zr);

		xr /= rij;
		yr /= rij;
		zr /= rij;
		    
		float fc = max((float)0, info.aij * (1 - rij / info.rc));
		float wr = max((float)0, 1 - rij / info.rc);
		float wd = wr * wr;

		float rdotv = xr * (info.xv[i] - info.xv[j]) + yr * (info.yv[i] - info.yv[j]) + zr * (info.zv[i] - info.zv[j]);
		float gij = info.rsamples[sr + c] * info.invsqrtdt;
		
		float xf = (fc - info.gamma * wd * rdotv + info.sigma * wr * gij) * xr;
		float yf = (fc - info.gamma * wd * rdotv + info.sigma * wr * gij) * yr;
		float zf = (fc - info.gamma * wd * rdotv + info.sigma * wr * gij) * zr;

		assert(!isnan(xf));
		assert(!isnan(yf));
		assert(!isnan(zf));

		info.xa[i] += xf;
		info.ya[i] += yf;
		info.za[i] += zf;

		info.xa[j] -= xf;
		info.ya[j] -= yf;
		info.za[j] -= zf;
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

__global__ void _fc(const int xoffset, const int yoffset, const int zoffset, int * consumed)
{
    if (blockIdx.x % 2 != xoffset ||
	blockIdx.y % 2 != yoffset ||
	blockIdx.z % 2 != zoffset)
	return;

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
	    
	    _ccfc(s1, e1, s2, e2, cid1 == cid2, offsetrsamples);
	}
    }
}

#include <unistd.h>

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
   		     float * const _rsamples, const int nsamples)
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
	
    int * consumed = NULL, * dconsumed = NULL;
    cudaHostAlloc((void **)&consumed, sizeof(int), cudaHostAllocMapped);
    assert(consumed != NULL);
    *consumed = 0;
    cudaHostGetDevicePointer(&dconsumed, consumed, 0);
	
    int nx = (int)ceil(XL / rc);
    int ny = (int)ceil(YL / rc);
    int nz = (int)ceil(ZL / rc);
    const int ncells = nx * ny * nz;
    
    device_vector<int> starts(ncells + 1);
    
    device_vector<float> xp(_xp, _xp + np), yp(_yp, _yp + np), zp(_zp, _zp + np),
	xv(_xv, _xv + np), yv(_yv, _yv + np), zv(_zv, _zv + np), rsamples(_rsamples, _rsamples + nsamples);

    device_vector<float> xa(np), ya(np), za(np);
    fill(xa.begin(), xa.end(), 0);
    fill(ya.begin(), ya.end(), 0);
    fill(za.begin(), za.end(), 0);
    
    InfoDPD c;
    c.nx = nx;
    c.ny = ny;
    c.nz = nz;
    c.np = np;
    c.nsamples = nsamples;
    c.XL = XL;
    c.YL = YL;
    c.ZL = ZL;
    c.xstart = -XL * 0.5; 
    c.ystart = -YL * 0.5; 
    c.zstart = -ZL * 0.5; 
    c.rc = rc;
    c.aij = aij;
    c.gamma = gamma;
    c.sigma = sigma;
    c.invsqrtdt = invsqrtdt;
    c.xp = _ptr(xp);
    c.yp = _ptr(yp);
    c.zp = _ptr(zp);
    c.xv = _ptr(xv);
    c.yv = _ptr(yv);
    c.zv = _ptr(zv);
    c.xa = _ptr(xa);
    c.ya = _ptr(ya);
    c.za = _ptr(za);
    c.rsamples = _ptr(rsamples);
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

    for(int code = 0; code < 8; ++code)
    {
	_fc<<<dim3(c.nx, c.ny, c.nz), dim3(8, 8, 1)>>>(code & 1, (code >> 1) & 1, (code >> 2) & 1, dconsumed);
	
	CUDA_CHECK(cudaPeekAtLastError());
	CUDA_CHECK(cudaThreadSynchronize());

	if (*consumed >= nsamples)
	{
	    printf("done with code %d: consumed: %d\n", code, *consumed);
	    printf("not a nice situation.\n");
	    abort();
	}
    }

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
}