 #include <sys/stat.h>
#include <sys/types.h>

#include <cmath>

#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/count.h>

#include <../saru.cuh>

#include "io.h"
#include "halo-exchanger.h"
#include "wall-interactions.h"

static const int MARGIN = 12;
static const int VPD = 256;

namespace SolidWallsKernel
{
    texture<float, 3, cudaReadModeElementType> texSDF;

    __device__ float sdf(float x, float y, float z, const int L)
    {
	float p[3] = {x, y, z};
	
	float texcoord[3];
	for(int c = 0; c < 3; ++c)
	{
	    texcoord[c] = (p[c] + L / 2 + MARGIN) / (L + 2 * MARGIN);

	    assert(texcoord[c] >= 0 && texcoord[c] <= 1);
	}
	
	return tex3D(texSDF, texcoord[0], texcoord[1], texcoord[2]);
    }

    __device__ float3 grad_sdf(float x, float y, float z)
    {
	const float p[3] = {x, y, z};
	
	float tc[3];
	for(int c = 0; c < 3; ++c)
	{
	    tc[c] = (p[c] + L / 2 + MARGIN) / (L + 2 * MARGIN);

	    if (!(tc[c] >= 0 && tc[c] <= 1))
	    {
		printf("oooooooooops wall-interactions: texture coordinate %f exceeds bounds [0, 1] for c %d\nincrease MARGIN or decrease timestep",
		       tc[c], c);
	    }
	    
	    assert(tc[c] >= 0 && tc[c] <= 1);
	}
	
	const float htw = 1. / VPD;
	const float factor = 1. / (2 * htw) * 1.f / (L * 2 + MARGIN);
	
	return make_float3(
	    factor * (tex3D(texSDF, tc[0] + htw, tc[1], tc[2]) - tex3D(texSDF, tc[0] - htw, tc[1], tc[2])),
	    factor * (tex3D(texSDF, tc[0], tc[1] + htw, tc[2]) - tex3D(texSDF, tc[0], tc[1] - htw, tc[2])),
	    factor * (tex3D(texSDF, tc[0], tc[1], tc[2] + htw) - tex3D(texSDF, tc[0], tc[1], tc[2] - htw))
	    );
    }
    
    __global__ void fill_keys(const Particle * const particles, const int n, const int L, int * const key)
    {
	assert(blockDim.x * gridDim.x >= n);

	const int pid = threadIdx.x + blockDim.x * blockIdx.x;

	if (pid >= n)
	    return;

	const Particle p = particles[pid];

	const float mysdf = sdf(p.x[0], p.x[1], p.x[2], L);
	key[pid] = (int)(mysdf >= 0) + (int)(mysdf > 3);
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

    __device__ bool handle_collision(float& x, float& y, float& z, float& u, float& v, float& w, const int L, const int rank, const double dt)
    {
	const float initial_sdf = sdf(x, y, z, L);
	
	if (initial_sdf < 0)
	    return false;
	
	const float xold = x - dt * u;
	const float yold = y - dt * v;
	const float zold = z - dt * w;

	if (sdf(xold, yold, zold, L) >= 0)
	{
	    //this is the worst case - it means that old position was bad already
	    //we need to rescue the particle, extracting it from the walls
	    for(int attempt = 0; attempt < 4; ++attempt)
	    {
		const float3 mygrad = grad_sdf(x, y, z);
		const float mysdf = sdf(x, y, z, L);
		
		for(int l = 0; l < 8; ++l)
		{
		    const float jump = pow(0.5f, l) * mysdf;
		    
		    x -= jump * mygrad.x;
		    y -= jump * mygrad.y;
		    z -= jump * mygrad.z;
		    
		    if (sdf(x, y, z, L) < 0)
		    {
			u  = -u;
			v  = -v;
			w  = -w;	    	
			
			return false;
		    }
		}
	    }
	    
	    printf("RANK %d bounce collision failed OLD: %f %f %f, sdf %e \nNEW: %f %f %f sdf %e\n", 
		   rank, 
		   xold, yold, zold, sdf(xold, yold, zold, L), 
		   x, y, z, sdf(x, y, z, L));
	    
	    return false;
	}
	
	float subdt = 0;
	    
	for(int i = 1; i < 8; ++i)
	{
	    const float tcandidate = subdt + dt / (1 << i);
	    const float xcandidate = xold + tcandidate * u;
	    const float ycandidate = yold + tcandidate * v;
	    const float zcandidate = zold + tcandidate * w;
	    
	    if (sdf(xcandidate, ycandidate, zcandidate, L) < 0)
		subdt = tcandidate;
	}
	
	const float lambda = 2 * subdt - dt;
	
	x = xold + lambda * u;
	y = yold + lambda * v;
	z = zold + lambda * w;
	
	u  = -u;
	v  = -v;
	w  = -w;	    
	
	if (sdf(x, y, z, L) >= 0)
	{
	    x = xold;
	    y = yold;
	    z = zold;
	    
	    assert(sdf(x, y, z, L) < 0);
	}

	return true;
    }

    __global__ void bounce(Particle * const particles, const int n, const int L, const int rank, const float dt)
    {
	assert(blockDim.x * gridDim.x >= n);

	const int pid = threadIdx.x + blockDim.x * blockIdx.x;

	if (pid >= n)
	    return;

	Particle p = particles[pid];

	for(int c = 0; c < 3; ++c)
	{
	    if (!(abs(p.x[c]) <= L/2 + MARGIN))
		printf("bounce: ooooooooops we have %f %f %f outside %d + %d\n", p.x[0], p.x[1], p.x[2], L/2, MARGIN);

	    assert(abs(p.x[c]) <= L/2 + MARGIN);
	}

	if (handle_collision(p.x[0], p.x[1], p.x[2], p.u[0], p.u[1], p.u[2], L, rank, dt))
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
	    assert(p.x[c] >= -L/2 - MARGIN);
	    assert(p.x[c] < L/2 + MARGIN);
	    base[c] = (int)(p.x[c] - (-L/2 - MARGIN));
	}

	const float xp = p.x[0], yp = p.x[1], zp = p.x[2];
	const float up = p.u[0], vp = p.u[1], wp = p.u[2];
	
	float xforce = 0, yforce = 0, zforce = 0;
	
	for(int code = 0; code < 27; ++code)
	{
	    const int xcid = base[0] + (code % 3) - 1;
	    const int ycid = base[1] + (code/3 % 3) - 1;
	    const int zcid = base[2] + (code/9 % 3) - 1;

	    if (xcid < 0 || xcid >= L + 2 * MARGIN ||
		ycid < 0 || ycid >= L + 2 * MARGIN ||
		zcid < 0 || zcid >= L + 2 * MARGIN )
		continue;
			    
	    const int cid = xcid + (L + 2 * MARGIN) * (ycid + (L + 2 * MARGIN) * zcid);
	    assert(cid >= 0 && cid < (L + 2 * MARGIN) * (L + 2 * MARGIN) * (L + 2 * MARGIN));

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
		const float argwr = max((float)0, 1 - rij);
		const float wr = powf(argwr, powf(0.5f, -VISCOSITY_S_LEVEL));

		const float xr = _xr * invrij;
		const float yr = _yr * invrij;
		const float zr = _zr * invrij;

		const float rdotv = 
		    xr * (up - 0) +
		    yr * (vp - 0) +
		    zr * (wp - 0);
		
		const float mysaru = saru(pid * nsolid + s, saru_tag1, saru_tag2);
	
		const float myrandnr = 3.464101615f * mysaru - 1.732050807f;
		 
		const float strength = aij * argwr + (- gamma * wr * rdotv + sigmaf * myrandnr) * wr;

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

template<int k>
struct Bspline
{
    template<int i>
    static float eval(float x)
	{
	    return
		(x - i) / (k - 1) * Bspline<k - 1>::template eval<i>(x) +
		(i + k - x) / (k - 1) * Bspline<k - 1>::template eval<i + 1>(x);
	}
};

template<>
struct Bspline<1>
{
    template <int i>
    static float eval(float x)
	{
	    return  (float)(i) <= x && x < (float)(i + 1);
	}
};

struct FieldSampler
{
    float * data, extent[3];
    int N[3];

    FieldSampler(const char * path)
	{
	    FILE * f = fopen(path, "rb");

#ifndef NDEBUG
	    int retval;
	    retval = 
#endif
		fscanf(f, "%f %f %f\n", extent + 0, extent + 1, extent + 2);
	    
	    assert(retval == 3);

#ifndef NDEBUG
	    retval = 
#endif
		fscanf(f, "%d %d %d\n", N + 0, N + 1, N + 2);
	
	    assert(retval == 3);
	    
	    const int nvoxels = N[0] * N[1] * N[2];
	    data = new float[nvoxels];

#ifndef NDEBUG	    
	    retval = 
#endif
		fread(data, sizeof(float), nvoxels, f);
	    assert(retval == nvoxels);

	    int nvoxels_solvent = 0;
	    for(int i = 0; i < nvoxels; ++i)
		nvoxels_solvent += (data[i] < 0);

	    fclose(f);
	}
    
    void sample(const float start[3], const float spacing[3], const int nsize[3], float * const output, const float amplitude_rescaling) 
	{
	    Bspline<4> bsp;

	    for(int iz = 0; iz < nsize[2]; ++iz)
		for(int iy = 0; iy < nsize[1]; ++iy)
		    for(int ix = 0; ix < nsize[0]; ++ix)
		    {
			const float x[3] = {
			    start[0] + (ix  + 0.5f) * spacing[0] - 0.5f,
			    start[1] + (iy  + 0.5f) * spacing[1] - 0.5f,
			    start[2] + (iz  + 0.5f) * spacing[2] - 0.5f
			};

			int anchor[3];
			for(int c = 0; c < 3; ++c)
			    anchor[c] = (int)floor(x[c]);
			
			float w[3][4];
			for(int c = 0; c < 3; ++c)
			    for(int i = 0; i < 4; ++i)
				w[c][i] = bsp.eval<0>(x[c] - (anchor[c] - 1 + i) + 2);
			
			float tmp[4][4];
			for(int sz = 0; sz < 4; ++sz)
			    for(int sy = 0; sy < 4; ++sy)
			    {
				float s = 0;
				
				for(int sx = 0; sx < 4; ++sx)
				{
				    const int l[3] = {sx, sy, sz};

				    int g[3];
				    for(int c = 0; c < 3; ++c)
					g[c] = max(0, min(N[c] - 1, l[c] - 1 + anchor[c]));

				    s += w[0][sx] * data[g[0] + N[0] * (g[1] + N[1] * g[2])];
				}

				tmp[sz][sy] = s;
			    }

			float partial[4];
			for(int sz = 0; sz < 4; ++sz)
			{
			    float s = 0;

			    for(int sy = 0; sy < 4; ++sy)
				s += w[1][sy] * tmp[sz][sy];

			    partial[sz] = s;
			}

			float val = 0;
			for(int sz = 0; sz < 4; ++sz)
			    val += w[2][sz] * partial[sz];
					
			output[ix + nsize[0] * (iy + nsize[1] * iz)] = val * amplitude_rescaling;
		    }
	}

    ~FieldSampler()
	{
	    delete [] data;
	}
};

ComputeInteractionsWall::ComputeInteractionsWall(MPI_Comm cartcomm, const int L, Particle* const p, 
						 const int n, int& nsurvived):
    cartcomm(cartcomm), L(L), arrSDF(NULL), solid(NULL), solid_size(0), cells(L+ 2 * MARGIN)
{
    MPI_CHECK( MPI_Comm_rank(cartcomm, &myrank));
    MPI_CHECK( MPI_Cart_get(cartcomm, 3, dims, periods, coords) );
    
    float * field = new float[VPD * VPD * VPD];

    FieldSampler sampler("sdf.dat");


#ifndef NDEBUG	
    assert(fabs(dims[0] / (double) dims[1] - sampler.extent[0] / (double)sampler.extent[1]) < 1e-5);
    assert(fabs(dims[0] / (double) dims[2] - sampler.extent[0] / (double)sampler.extent[2]) < 1e-5);
#endif
        
    {
       	float start[3], spacing[3];
	for(int c = 0; c < 3; ++c)
	{
	    start[c] = (coords[c] * L - MARGIN) / (float)(dims[c] * L) * sampler.N[c];
	    spacing[c] =  sampler.N[c] * (L + 2 * MARGIN) / (float)(dims[c] * L * VPD) ;
	}
	
	int size[3] = {VPD, VPD, VPD};

	const float amplitude_rescaling = (L + 2 * MARGIN) / (sampler.extent[0] / dims[0]) ;
	sampler.sample(start, spacing, size, field, amplitude_rescaling);
    }

    if (hdf5field_dumps)
    {
	float * walldata = new float[L * L * L];

	float start[3], spacing[3];
	for(int c = 0; c < 3; ++c)
	{
	    start[c] = coords[c] * L / (float)(dims[c] * L) * sampler.N[c];
	    spacing[c] = sampler.N[c] / (float)(dims[c] * L) ;
	}
	
	int size[3] = {L, L, L};

	const float amplitude_rescaling = L / (sampler.extent[0] / dims[0]);
	sampler.sample(start, spacing, size, walldata, amplitude_rescaling);

	H5FieldDump dump(cartcomm);
	dump.dump_scalarfield(walldata, "wall");

	delete [] walldata;
    }
    
    CUDA_CHECK(cudaPeekAtLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

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

    const int nbelt = thrust::count(keys.begin() + nsurvived, keys.end(), 1);
    printf("rank %d belt is %d -> %.2f%%\n", myrank, nbelt, nbelt * 100. / n);

    thrust::device_vector<Particle> solid_local(thrust::device_ptr<Particle>(p + nsurvived), thrust::device_ptr<Particle>(p + nsurvived + nbelt));
  
    //can't use halo-exchanger class because of MARGIN
    //HaloExchanger halo(cartcomm, L, 666);
    //SimpleDeviceBuffer<Particle> solid_remote;
    //halo.exchange(thrust::raw_pointer_cast(&solid_local[0]), solid_local.size(), solid_remote);

    SimpleDeviceBuffer<Particle> solid_remote;

    {
	thrust::host_vector<Particle> local = solid_local;

	int dstranks[26], remsizes[26], recv_tags[26];
	for(int i = 0; i < 26; ++i)
	{
	    const int d[3] = { (i + 2) % 3 - 1, (i / 3 + 2) % 3 - 1, (i / 9 + 2) % 3 - 1 };
	    
	    recv_tags[i] = (2 - d[0]) % 3 + 3 * ((2 - d[1]) % 3 + 3 * ((2 - d[2]) % 3));
	    
	    int coordsneighbor[3];
	    for(int c = 0; c < 3; ++c)
		coordsneighbor[c] = coords[c] + d[c];
	    
	    MPI_CHECK( MPI_Cart_rank(cartcomm, coordsneighbor, dstranks + i) );
	}

	//send local counts - receive remote counts
	{
	    for(int i = 0; i < 26; ++i)
		remsizes[i] = -1;

	    MPI_Request reqrecv[26];
	    for(int i = 0; i < 26; ++i)
		MPI_CHECK( MPI_Irecv(remsizes + i, 1, MPI_INTEGER, dstranks[i], 123 + recv_tags[i], cartcomm, reqrecv + i) );
	    
	    const int localsize = local.size();
	    
	    MPI_Request reqsend[26];
	    for(int i = 0; i < 26; ++i)
		MPI_CHECK( MPI_Isend(&localsize, 1, MPI_INTEGER, dstranks[i], 123 + i, cartcomm, reqsend + i) );
	    
	    MPI_Status statuses[26];
	    MPI_CHECK( MPI_Waitall(26, reqrecv, statuses) );    
	    MPI_CHECK( MPI_Waitall(26, reqsend, statuses) );  

	    for(int i = 0; i < 26; ++i)
		assert(remsizes[i] >= 0);
	}

	std::vector<Particle> remote[26];

	//send local data - receive remote data
	{
	    for(int i = 0; i < 26; ++i)
		remote[i].resize(remsizes[i]);

	    MPI_Request reqrecv[26];
	    for(int i = 0; i < 26; ++i)
		MPI_CHECK( MPI_Irecv(remote[i].data(), remote[i].size() * 6, MPI_FLOAT, dstranks[i], 321 + recv_tags[i], cartcomm, reqrecv + i) );

	    MPI_Request reqsend[26];
	    for(int i = 0; i < 26; ++i)
		MPI_CHECK( MPI_Isend(local.data(), local.size() * 6, MPI_FLOAT, dstranks[i], 321 + i, cartcomm, reqsend + i) );
	    
	    MPI_Status statuses[26];
	    MPI_CHECK( MPI_Waitall(26, reqrecv, statuses) );    
	    MPI_CHECK( MPI_Waitall(26, reqsend, statuses) );
	}

	//select particles within my region [-L / 2 - MARGIN, +L / 2 + MARGIN]
	std::vector<Particle> selected;
	for(int i = 0; i < 26; ++i)
	{
	    int d[3] = { (i + 2) % 3 - 1, (i / 3 + 2) % 3 - 1, (i / 9 + 2) % 3 - 1 };

	    for(int j = 0; j < remote[i].size(); ++j)
	    {
		Particle p = remote[i][j];

		for(int c = 0; c < 3; ++c)
		    p.x[c] += d[c] * L;

		bool inside = true;

		for(int c = 0; c < 3; ++c)
		    inside &= p.x[c] >= -L / 2 - MARGIN && p.x[c] < L / 2 + MARGIN;

		if (inside)
		    selected.push_back(p);
	    }
	}

	solid_remote.resize(selected.size());
	CUDA_CHECK(cudaMemcpy(solid_remote.data, selected.data(), sizeof(Particle) * solid_remote.size, cudaMemcpyHostToDevice));
    }

    printf("rank %d is receiving extra %d\n", myrank, solid_remote.size);
    
    solid_size = solid_local.size() + solid_remote.size;

    CUDA_CHECK(cudaMalloc(&solid, sizeof(Particle) * solid_size));
    CUDA_CHECK(cudaMemcpy(solid, thrust::raw_pointer_cast(&solid_local[0]), sizeof(Particle) * solid_local.size(), cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(solid + solid_local.size(), solid_remote.data, sizeof(Particle) * solid_remote.size, cudaMemcpyDeviceToDevice));
        
    if (solid_size > 0)
	SolidWallsKernel::zero_velocity<<< (solid_size + 127) / 128, 128>>>(solid, solid_size);

    if (solid_size > 0)
	cells.build(solid, solid_size);

    {
	const int n = solid_local.size();

	Particle * phost = new Particle[n];

	CUDA_CHECK(cudaMemcpy(phost, thrust::raw_pointer_cast(&solid_local[0]), sizeof(Particle) * n, cudaMemcpyDeviceToHost));

	H5PartDump solid_dump("solid-walls.h5part", cartcomm, L);
	solid_dump.dump(phost, n);

	delete [] phost;
    }

    CUDA_CHECK(cudaPeekAtLastError());
}

void ComputeInteractionsWall::bounce(Particle * const p, const int n)
{
    if (n > 0)
	SolidWallsKernel::bounce<<< (n + 127) / 128, 128>>>(p, n, L, myrank, dt);
    
    CUDA_CHECK(cudaPeekAtLastError());
}

void ComputeInteractionsWall::interactions(const Particle * const p, const int n, Acceleration * const acc,
			      const int * const cellsstart, const int * const cellscount, int& saru_tag)
{
    //cellsstart and cellscount IGNORED for now
    
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
