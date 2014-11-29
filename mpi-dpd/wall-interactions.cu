#include <cmath>

#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/count.h>

#include <../saru.cuh>

#include "halo-exchanger.h"

#include "wall-interactions.h"

namespace SolidWallsKernel
{
    texture<float, 3, cudaReadModeElementType> texSDF;

    __device__ float sdf(float x, float y, float z, const int L)
    {
	float p[3] = {x, y, z};
	
	float texcoord[3];
	for(int c = 0; c < 3; ++c)
	{
	    texcoord[c] = (p[c] - (-L/2 - 1)) / (L + 2);
	    assert(texcoord[c] >= 0 && texcoord[c] <= 1);
	}
	
	return tex3D(texSDF, texcoord[0], texcoord[1], texcoord[2]);
    }
    
    __global__ void fill_keys(const Particle * const particles, const int n, const int L, int * const key)
    {
	assert(blockDim.x * gridDim.x >= n);

	const int pid = threadIdx.x + blockDim.x * blockIdx.x;

	if (pid >= n)
	    return;

	const Particle p = particles[pid];

	key[pid] = (int)(sdf(p.x[0], p.x[1], p.x[2], L) > 0);
    }

    __global__ void zero_velocity(Particle * const dst, const int n)
    {
	assert(blockDim.x * gridDim.x >= n);

	const int pid = threadIdx.x + blockDim.x * blockIdx.x;

	if (pid >= n)
	    return;

	Particle p = dst[pid];

	for(int c = 0; c < 3; ++c)
	    p.u[c] = 0;

	dst[pid] = p;
    }

    __device__ bool handle_collision(float& x, float& y, float& z, float& u, float& v, float& w, /*float& dt,*/ const int L)
    {
	if (sdf(x, y, z, L) <= 0)
	    return false;

	const float xold = x - dt * u;
	const float yold = y - dt * v;
	const float zold = z - dt * w;

	float t = 0;

	for(int i = 1; i < 8; ++i)
	{
	    const float tcandidate = t + dt / (1 << i);
	    const float xcandidate = xold + tcandidate * u;
	    const float ycandidate = yold + tcandidate * v;
	    const float zcandidate = zold + tcandidate * w;

	    if (sdf(xcandidate, ycandidate, zcandidate, L) <= 0)
		t = tcandidate;
	}

	const float lambda = 2 * t - dt;

	x = xold + lambda * u;
	y = yold + lambda * v;
	z = zold + lambda * w;

	u  = -u;
	v  = -v;
	w  = -w;
	//dt = dt - t;

	return true;
    }

    __global__ void bounce(Particle * const particles, const int n, const int L) //, const float dt)
    {
	assert(blockDim.x * gridDim.x >= n);

	const int pid = threadIdx.x + blockDim.x * blockIdx.x;

	if (pid >= n)
	    return;

	Particle p = particles[pid];

	for(int c = 0; c < 3; ++c)
	    assert(abs(p.x[c]) <= L/2 + 1);

	if (handle_collision(p.x[0], p.x[1], p.x[2], p.u[0], p.u[1], p.u[2], L))
	    particles[pid] = p;
    }

    __global__ void interactions(const Particle * const particles, const int np, Acceleration * const acc,
				 const int * const starts, const int * const counts, const int L,
				 const Particle * const solid, const int nsolid, const int saru_tag1, const int saru_tag2,
				 const float aij, const float gamma, const float sigmaf)
    {
	assert(blockDim.x * gridDim.x >= np);

       	const int pid = threadIdx.x + blockDim.x * blockIdx.x;

	if (pid >= np)
	    return;

	Particle p = particles[pid];
	
	int base[3];
	for(int c = 0; c < 3; ++c)
	{
	    assert(p.x[c] >= -L/2 - 1);
	    base[c] = (int)(p.x[c] - (-L/2 -1));
	}

	const float xp = p.x[0], yp = p.x[1], zp = p.x[2];
	const float up = p.u[0], vp = p.u[1], wp = p.u[2];
	
	float xforce = 0, yforce = 0, zforce = 0;
	
	for(int code = 0; code < 27; ++code)
	{
	    const int xcid = base[0] + (code % 3) - 1;
	    const int ycid = base[1] + (code/3 % 3) - 1;
	    const int zcid = base[2] + (code/9 % 3) - 1;

	    if (xcid < 0 || xcid >= L + 2 ||
		ycid < 0 || ycid >= L + 2 ||
		zcid < 0 || zcid >= L + 2 )
		continue;
			    
	    const int cid = xcid + (L + 2) * (ycid + (L + 2) * zcid);
	    assert(cid >= 0 && cid < (L + 2) * (L + 2) * (L + 2));

	    const int start = starts[cid];
	    const int stop = start + counts[cid];

	    assert(start >= 0 && stop <= nsolid && start <= stop);

	    for(int s = start; s < stop; ++s)
	    {
		const float xq = solid[s].x[0];
		const float yq = solid[s].x[1];
		const float zq = solid[s].x[2];
		
	    	const float _xr = xp - xq;
		const float _yr = yp - yq;
		const float _zr = zp - zq;
		
		const float rij2 = _xr * _xr + _yr * _yr + _zr * _zr;
		
		const float invrij = rsqrtf(rij2);
		 
		const float rij = rij2 * invrij;
		const float wr = max((float)0, 1 - rij);
		
		const float xr = _xr * invrij;
		const float yr = _yr * invrij;
		const float zr = _zr * invrij;

		const float rdotv = 
		    xr * (up - 0) +
		    yr * (vp - 0) +
		    zr * (wp - 0);
		
		const float mysaru = saru(pid * nsolid + s, saru_tag1, saru_tag2);
	
		const float myrandnr = 3.464101615f * mysaru - 1.732050807f;
		 
		const float strength = (aij - gamma * wr * rdotv + sigmaf * myrandnr) * wr;

		xforce += strength * xr;
		yforce += strength * yr;
		zforce += strength * zr;
	    }
	}

	acc[pid].a[0] += xforce;
	acc[pid].a[1] += yforce;
	acc[pid].a[2] += zforce;

	for(int c = 0; c < 3; ++c)
	    assert(!isnan(acc[pid].a[c]));
    }
}

float smoothstep(float edge0, float edge1, float x)
{
    // Scale, and clamp x to 0..1 range
    x = min(1.f, max(0.f, (x - edge0)/(edge1 - edge0)));
    // Evaluate polynomial
    return x*x*x*(x*(x*6 - 15) + 10);
}

ComputeInteractionsWall::ComputeInteractionsWall(MPI_Comm cartcomm, const int L, Particle* const p, 
						 const int n, int& nsurvived):
    cartcomm(cartcomm), L(L), arrSDF(NULL), solid(NULL), solid_size(0), cells(L+2)
{
    MPI_CHECK( MPI_Comm_rank(cartcomm, &myrank));
    MPI_CHECK( MPI_Cart_get(cartcomm, 3, dims, periods, coords) );
    
    const int VPD = 64;

    float * field = new float[VPD * VPD * VPD];

    const double h = (L + 2) / (double)(VPD - 2);

#if 1

    //bifurcation
    const float r_cyl = 0.085 * L * dims[1];
    const float r2_cyl = r_cyl * r_cyl;
    
    static const int nh = 75 * 2;
    const float dh = 1./ (nh - 1);
    std::vector<float> xpts, ypts, zpts;
    
    for(int pass = 0; pass < 2; ++pass)
	for(int i = 0; i < nh; ++i)
	{
	    const float S0 = 0.15;
	    const float S1 = 0.5;
	    const float S2 = 0.85;
	    const float X0 = (S1 + S2) * 0.5 * dims[0] * L;
	    const float X1 = S2 * dims[0] * L;
	    const float Y0 = 0.5 * dims[1] * L;
	    const float Y1 = 0.8 * dims[1] * L;
	    const float Z0 = 0.5 * dims[2] * L;
	    const float Z1 = (0.5 + 1 * 0.35) * dims[2] * L;

	    const float lambda = i * dh;
	    
	    float x = lambda / S1 * X0;
	    if (lambda > S2)
		x = X0 + (lambda - S2) / (1 - S2) * (dims[0] * L - X0);
	    else
		if (lambda > S1)
		    x = X0 + sin( (lambda - S1) / (S2 - S1) * 2 * 3.1415) * (X1 - X0);
	   
	    const float y = Y0 + (2 * pass -1) * smoothstep(S0, S1, min(lambda, 1 - lambda)) * (Y1 - Y0);

	    float z = Z0;

	    if (lambda >= S1 && lambda < S2)
		z = 0.5 * (Z0 + Z1) - 0.5 * (Z1 - Z0) * cos( (lambda - S1) / (S2 - S1) * 2 * 3.1415);

	    xpts.push_back(x);
	    ypts.push_back(y);
	    zpts.push_back(z);
	}

    const int npts = xpts.size();
    
    for(int iz = 0; iz < VPD; ++iz)
	for(int iy = 0; iy < VPD; ++iy)
	    for(int ix = 0; ix < VPD; ++ix)
	    {
		const float x = coords[0] * L - 1 + (ix + 0.5) * h;
		const float y = coords[1] * L - 1 + (iy + 0.5) * h;
		const float z = coords[2] * L - 1 + (iz + 0.5) * h;

		float val = 1;
		for(int i = 0; i < npts && val == 1; ++i)
		{
		    const float dx = x - xpts[i];
		    const float dy = y - ypts[i];
		    const float dz = z - zpts[i];

		    if (dx * dx + dy * dy + dz * dz < r2_cyl)
			val = -1;
		}
				
		field[ix + VPD * (iy + VPD * iz)] = val;
	    }
#else

    const float y_cyl = 0.5 * L * dims[1];
    const float z_cyl = 0.5 * L * dims[2];
    const float r_cyl = 0.125 * L * dims[1];
    
    //cylindrical pipe
    for(int iz = 0; iz < VPD; ++iz)
	for(int iy = 0; iy < VPD; ++iy)
	    for(int ix = 0; ix < VPD; ++ix)
	    {
		//const float x = coords[0] * L - 1 + (ix + 0.5) * h;
		const float y = coords[1] * L - 1 + (iy + 0.5) * h;
		const float z = coords[2] * L - 1 + (iz + 0.5) * h;

		const float r = sqrt(pow(y - y_cyl, 2) + pow(z - z_cyl, 2));
		
		field[ix + VPD * (iy + VPD * iz)] = r - r_cyl;
	    }
#endif
    
    cudaChannelFormatDesc fmt = cudaCreateChannelDesc<float>();
    CUDA_CHECK(cudaMalloc3DArray (&arrSDF, &fmt, make_cudaExtent(VPD, VPD, VPD)));

    cudaMemcpy3DParms copyParams = {0};
    copyParams.srcPtr   = make_cudaPitchedPtr((void *)field, VPD * sizeof(float), VPD, VPD);
    copyParams.dstArray = arrSDF;
    copyParams.extent   = make_cudaExtent(VPD, VPD, VPD);
    copyParams.kind     = cudaMemcpyHostToDevice;
    CUDA_CHECK(cudaMemcpy3D(&copyParams));

    for(int i = 0; i < 3; ++i)
	SolidWallsKernel::texSDF.addressMode[i] = cudaAddressModeClamp;

    SolidWallsKernel::texSDF.normalized = true;
    SolidWallsKernel::texSDF.filterMode = cudaFilterModeLinear;
    SolidWallsKernel::texSDF.addressMode[0] = cudaAddressModeClamp;
    SolidWallsKernel::texSDF.addressMode[1] = cudaAddressModeClamp;
    SolidWallsKernel::texSDF.addressMode[2] = cudaAddressModeClamp;
		
    CUDA_CHECK(cudaBindTextureToArray(SolidWallsKernel::texSDF, arrSDF, fmt));

    delete [] field;

    thrust::device_vector<int> keys(n);

    SolidWallsKernel::fill_keys<<< (n + 127) / 128, 128 >>>(p, n, L, thrust::raw_pointer_cast(&keys[0]));
    CUDA_CHECK(cudaPeekAtLastError());
    
    thrust::sort_by_key(keys.begin(), keys.end(), thrust::device_ptr<Particle>(p));

    nsurvived = thrust::count(keys.begin(), keys.end(), 0);
    assert(nsurvived <= n);
    
    printf("rank %d nsurvived is %d -> %.2f%%\n", myrank, nsurvived, nsurvived * 100. /n);


    thrust::device_vector<Particle> solid_local(thrust::device_ptr<Particle>(p + nsurvived), thrust::device_ptr<Particle>(p + n));
  
    HaloExchanger halo(cartcomm, L);

    SimpleDeviceBuffer<Particle> solid_remote = halo.exchange(thrust::raw_pointer_cast(&solid_local[0]), solid_local.size());

    printf("receiving extra %d\n", solid_remote.size);
    
    solid_size = solid_local.size() + solid_remote.size;

    CUDA_CHECK(cudaMalloc(&solid, sizeof(Particle) * solid_size));
    CUDA_CHECK(cudaMemcpy(solid, thrust::raw_pointer_cast(&solid_local[0]), sizeof(Particle) * solid_local.size(), cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(solid + solid_local.size(), solid_remote.data, sizeof(Particle) * solid_remote.size, cudaMemcpyDeviceToDevice));
        
    SolidWallsKernel::zero_velocity<<< (solid_size + 127) / 128, 128>>>(solid, solid_size);

    if (solid_size > 0)
	cells.build(solid, solid_size);

    {
	const int n = solid_local.size();

	Particle * phost = new Particle[n];

	CUDA_CHECK(cudaMemcpy(phost, thrust::raw_pointer_cast(&solid_local[0]), sizeof(Particle) * n, cudaMemcpyDeviceToHost));

	H5PartDump solid_dump /*ha ha ha*/("solid-walls.h5part", cartcomm, L);
	solid_dump.dump(phost, n);

	delete [] phost;
    }
}

void ComputeInteractionsWall::bounce(Particle * const p, const int n)
{
    if (n > 0)
	SolidWallsKernel::bounce<<< (n + 127) / 128, 128>>>(p, n, L);
    
    CUDA_CHECK(cudaPeekAtLastError());
}

void ComputeInteractionsWall::interactions(const Particle * const p, const int n, Acceleration * const acc,
			      const int * const cellsstart, const int * const cellscount, int& saru_tag)
{
    if (n > 0 && solid_size > 0)
	SolidWallsKernel::interactions<<< (n + 127) / 128, 128>>>(p, n, acc, cells.start, cells.count, L,
								  solid, solid_size, saru_tag, myrank, aij, gammadpd, sigmaf);

    CUDA_CHECK(cudaPeekAtLastError());

    ++saru_tag;
}

ComputeInteractionsWall::~ComputeInteractionsWall()
{
    CUDA_CHECK(cudaUnbindTexture(SolidWallsKernel::texSDF));
    CUDA_CHECK(cudaFreeArray(arrSDF));
}
