
/*
 *  containers.cu
 *  Part of CTC/mpi-dpd/
 *
 *  Created and authored by Diego Rossinelli on 2014-12-05.
 *  Copyright 2015. All rights reserved.
 *
 *  Users are NOT authorized
 *  to employ the present software for their own publications
 *  before getting a written permission from the author of this file.
 */

#include <sys/stat.h>

#include <rbc-cuda.h>

#include "containers.h"
#include "io.h"
#include "ctc.h"

int (*CollectionRBC::indices)[3] = NULL, CollectionRBC::ntriangles = -1, CollectionRBC::nvertices = -1;

int (*CollectionCTC::indices)[3] = NULL, CollectionCTC::ntriangles = -1, CollectionCTC::nvertices = -1;

namespace ParticleKernels
{
    __global__ void update_stage1(Particle * p, Acceleration * a, int n, float dt,
				  const float driving_acceleration, const bool check = true)
    {
	assert(blockDim.x * gridDim.x >= n);

	const int pid = threadIdx.x + blockDim.x * blockIdx.x;

	if (pid >= n)
	    return;

	for(int c = 0; c < 3; ++c)
	{
	    assert(!isnan(p[pid].x[c]));
	    assert(!isnan(p[pid].u[c]));
	    assert(!isnan(a[pid].a[c]));
	}

	for(int c = 0; c < 3; ++c)
	    p[pid].u[c] += (a[pid].a[c] + (c == 0 ? driving_acceleration : 0)) * dt * 0.5;

	for(int c = 0; c < 3; ++c)
	    p[pid].x[c] += p[pid].u[c] * dt;

#ifndef NDEBUG
	const int L[3] = { XSIZE_SUBDOMAIN, YSIZE_SUBDOMAIN, ZSIZE_SUBDOMAIN };

	if (check)
	    for(int c = 0; c < 3; ++c)
	    {
		assert(p[pid].x[c] >= -L[c] -L[c]/2);
		assert(p[pid].x[c] <= +L[c] +L[c]/2);
	    }
#endif
    }

    __global__ void update_stage2_and_1(float2 * const pdata, const float * const adata,
					const int nparticles, const float dt, const float driving_acceleration)
    {
	
#if !defined(__CUDA_ARCH__)
#warning __CUDA_ARCH__ not defined! assuming 350
#define _ACCESS(x) __ldg(x)
#elif __CUDA_ARCH__ >= 350
#define _ACCESS(x) __ldg(x)
#else
#define _ACCESS(x) (*(x))
#endif

	enum { NWARPS = 4, STRIDE = 32 };

	assert(blockDim.x * blockDim.y * gridDim.x >= nparticles && blockDim.x == 32 && blockDim.y == NWARPS);

	__shared__ volatile float shxv[NWARPS][STRIDE * 6];

	const int pidbase = 32 * (threadIdx.y + NWARPS * blockIdx.x);
	const int nlocalparticles = min(nparticles - pidbase, 32);
	const int nwords = nlocalparticles * 3;

	const int base = 3 * pidbase;
	const int wid = threadIdx.y;
	const int tid = threadIdx.x;

	float ax, ay, az;

	const bool valid = tid < nlocalparticles;

	if (valid)
	{
	    const int entry = 3 * (pidbase + tid);
	    ax = _ACCESS(adata + entry);
	    ay = _ACCESS(adata + entry + 1);
	    az = _ACCESS(adata + entry + 2);
	}

	float2 tmp2[3] = {0, 0, 0};

#pragma unroll 3
	for(int c = 0; c < 3; ++c)
	    if (tid + 32 * c < nwords)
		tmp2[c] = _ACCESS(pdata + base + tid + 32 * c);

#pragma unroll 3
	for(int c = 0; c < 3; ++c)
	{
	    const int entry = 2 * (tid + 32 * c); 
	    const int lpid = entry / 6;
	    const int c0 = entry % 6;
	    const int c1 = (entry + 1) % 6;
	    
	    shxv[wid][(c0 % 3) + 3 * (lpid + STRIDE * (c0 >= 3))] = tmp2[c].x;
	    shxv[wid][(c1 % 3) + 3 * (lpid + STRIDE * (c1 >= 3))] = tmp2[c].y;
	}

	if (valid)
	{
	    const int entry0 = 3 * tid;
	    const int entry1 = 3 * (tid + STRIDE);
	    
	    float x = shxv[wid][entry0];
	    float y = shxv[wid][entry0 + 1];
	    float z = shxv[wid][entry0 + 2];
	    float u = shxv[wid][entry1];
	    float v = shxv[wid][entry1 + 1];
	    float w = shxv[wid][entry1 + 2];

	    u += (ax + driving_acceleration) * dt;
	    v += ay * dt;
	    w += az * dt;

	    x += u * dt;
	    y += v * dt;
	    z += w * dt;
	    
	    shxv[wid][entry0] = x;
	    shxv[wid][entry0 + 1] = y;
	    shxv[wid][entry0 + 2] = z;
	    shxv[wid][entry1] = u;
	    shxv[wid][entry1 + 1] = v;
	    shxv[wid][entry1 + 2] = w;
	}

#pragma unroll 3
	for(int c = 0; c < 3; ++c)
	{
	    const int entry = 2 * (tid + 32 * c); 
	    const int lpid = entry / 6;
	    const int c0 = entry % 6;
	    const int c1 = (entry + 1) % 6;
	    
	    tmp2[c].x = shxv[wid][(c0 % 3) + 3 * (lpid + STRIDE * (c0 >= 3))];
	    tmp2[c].y = shxv[wid][(c1 % 3) + 3 * (lpid + STRIDE * (c1 >= 3))];
	}

#pragma unroll 3
	for(int c = 0; c < 3; ++c)
	    if (tid + 32 * c < nwords)
		pdata[base + tid + 32 * c] = tmp2[c];
    }

    __global__ void clear_velocity(Particle * const p, const int n)
    {
	assert(blockDim.x * gridDim.x >= n);

	const int pid = threadIdx.x + blockDim.x * blockIdx.x;

	if (pid >= n)
	    return;

	for(int c = 0; c < 3; ++c)
	    p[pid].u[c] = 0;
    }
}

ParticleArray::ParticleArray(vector<Particle> ic)
{
    resize(ic.size());

    CUDA_CHECK(cudaMemcpy(xyzuvw.data, (float*) &ic.front(), sizeof(Particle) * ic.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(axayaz.data, 0, sizeof(Acceleration) * ic.size()));

//    CUDA_CHECK(cudaFuncSetCacheConfig(*ParticleKernels::update_stage2_and_1, cudaFuncCachePreferL1));
}

void ParticleArray::update_stage1(const float driving_acceleration, cudaStream_t stream)
{
    if (size)
	ParticleKernels::update_stage1<<<(xyzuvw.size + 127) / 128, 128, 0, stream>>>(
	    xyzuvw.data, axayaz.data, xyzuvw.size, dt, driving_acceleration , false);
}

void  ParticleArray::update_stage2_and_1(const float driving_acceleration, cudaStream_t stream)
{
    if (size)
	ParticleKernels::update_stage2_and_1<<<(xyzuvw.size + 127) / 128, dim3(32, 4), 0, stream>>>
	    ((float2 *)xyzuvw.data, (float *)axayaz.data, xyzuvw.size, dt, driving_acceleration);
}

void ParticleArray::resize(int n)
{
    size = n;

    // YTANG: need the array to be 32-padded for locally transposed storage of acceleration
    if ( n % 32 ) {
        xyzuvw.preserve_resize( n - n % 32 + 32 );
        axayaz.preserve_resize( n - n % 32 + 32 );
    }
    xyzuvw.resize(n);
    axayaz.resize(n);

    CUDA_CHECK(cudaMemset(axayaz.data, 0, sizeof(Acceleration) * size));
}

void ParticleArray::preserve_resize(int n)
{
    int oldsize = size;
    size = n;

    xyzuvw.preserve_resize(n);
    axayaz.preserve_resize(n);

    if (size > oldsize)
    	CUDA_CHECK(cudaMemset(axayaz.data + oldsize, 0, sizeof(Acceleration) * (size-oldsize)));
}

void ParticleArray::clear_velocity()
{
    if (size)
	ParticleKernels::clear_velocity<<<(xyzuvw.size + 127) / 128, 128 >>>(xyzuvw.data, xyzuvw.size);
}

void CollectionRBC::resize(const int count)
{
    ncells = count;

    ParticleArray::resize(count * get_nvertices());
}

void CollectionRBC::preserve_resize(const int count)
{
    ncells = count;

    ParticleArray::preserve_resize(count * get_nvertices());
}

struct TransformedExtent
{
    float com[3];
    float transform[4][4];
};

CollectionRBC::CollectionRBC(MPI_Comm cartcomm):
cartcomm(cartcomm), ncells(0)
{
    MPI_CHECK(MPI_Comm_rank(cartcomm, &myrank));
    MPI_CHECK( MPI_Cart_get(cartcomm, 3, dims, periods, coords) );

    CudaRBC::get_triangle_indexing(indices, ntriangles);
    CudaRBC::Extent extent;
    CudaRBC::setup(nvertices, extent);

    assert(extent.xmax - extent.xmin < XSIZE_SUBDOMAIN);
    assert(extent.ymax - extent.ymin < YSIZE_SUBDOMAIN);
    assert(extent.zmax - extent.zmin < ZSIZE_SUBDOMAIN);
}

void CollectionRBC::setup(const char * const path2ic)
{
    vector<TransformedExtent> allrbcs;

    if (myrank == 0)
    {
	//read transformed extent from file
	FILE * f = fopen(path2ic, "r");
	printf("READING FROM: <%s>\n", path2ic);
	bool isgood = true;

	while(isgood)
	{
	    float tmp[19];
	    for(int c = 0; c < 19; ++c)
	    {
		int retval = fscanf(f, "%f", tmp + c);

		isgood &= retval == 1;
	    }

	    if (isgood)
	    {
		TransformedExtent t;

		for(int c = 0; c < 3; ++c)
		    t.com[c] = tmp[c];

		int ctr = 3;
		for(int c = 0; c < 16; ++c, ++ctr)
		    t.transform[c / 4][c % 4] = tmp[ctr];

		allrbcs.push_back(t);
	    }
	}

	fclose(f);
    }

    if (myrank == 0)
	printf("Instantiating %d CELLs from...<%s>\n", (int)allrbcs.size(), path2ic);

    int allrbcs_count = allrbcs.size();
    MPI_CHECK(MPI_Bcast(&allrbcs_count, 1, MPI_INT, 0, cartcomm));

    allrbcs.resize(allrbcs_count);

    const int nfloats_per_entry = sizeof(TransformedExtent) / sizeof(float);
    assert( sizeof(TransformedExtent) % sizeof(float) == 0);

    MPI_CHECK(MPI_Bcast(&allrbcs.front(), nfloats_per_entry * allrbcs_count, MPI_FLOAT, 0, cartcomm));

    vector<TransformedExtent> good;

    const int L[3] = { XSIZE_SUBDOMAIN, YSIZE_SUBDOMAIN, ZSIZE_SUBDOMAIN };

    for(vector<TransformedExtent>::iterator it = allrbcs.begin(); it != allrbcs.end(); ++it)
    {
	bool inside = true;

	for(int c = 0; c < 3; ++c)
	    inside &= it->com[c] >= coords[c] * L[c] && it->com[c] < (coords[c] + 1) * L[c];

	if (inside)
	{
	    for(int c = 0; c < 3; ++c)
		it->transform[c][3] -= (coords[c] + 0.5) * L[c];

	    good.push_back(*it);
	}
    }

    resize(good.size());

    for(int i = 0; i < good.size(); ++i)
	_initialize((float *)(xyzuvw.data + get_nvertices() * i), good[i].transform);
}

void CollectionRBC::_initialize(float *device_xyzuvw, const float (*transform)[4])
{
    CudaRBC::initialize(device_xyzuvw, transform);
}

void CollectionRBC::remove(const int * const entries, const int nentries)
{
    std::vector<bool > marks(ncells, true);

    for(int i = 0; i < nentries; ++i)
	marks[entries[i]] = false;

    std::vector< int > survivors;
    for(int i = 0; i < ncells; ++i)
	if (marks[i])
	    survivors.push_back(i);

    const int nsurvived = survivors.size();

    SimpleDeviceBuffer<Particle> survived(get_nvertices() * nsurvived);

    for(int i = 0; i < nsurvived; ++i)
	CUDA_CHECK(cudaMemcpy(survived.data + get_nvertices() * i, data() + get_nvertices() * survivors[i],
			      sizeof(Particle) * get_nvertices(), cudaMemcpyDeviceToDevice));

    resize(nsurvived);

    CUDA_CHECK(cudaMemcpy(xyzuvw.data, survived.data, sizeof(Particle) * survived.size, cudaMemcpyDeviceToDevice));
}

void CollectionRBC::_dump(const char * const path2xyz, const char * const format4ply,
			  MPI_Comm comm, MPI_Comm cartcomm, const int ntriangles, const int ncells, const int nvertices, int (* const indices)[3],
			  Particle * const p, const Acceleration * const a, const int n, const int iddatadump)
{
    int ctr = iddatadump;
    const bool firsttime = ctr == 0;

    //we fused VV stages so we need to recover the state before stage 1
    for(int i = 0; i < n; ++i)
	for(int c = 0; c < 3; ++c)
	{
	    assert(!isnan(p[i].x[c]));
	    assert(!isnan(p[i].u[c]));
	    assert(!isnan(a[i].a[c]));

	    p[i].x[c] -= dt * p[i].u[c];
	    p[i].u[c] -= 0.5 * dt * a[i].a[c];
	}

    if (xyz_dumps)
	xyz_dump(comm, cartcomm, path2xyz, "cell-particles", p, n, !firsttime);

    char buf[200];
    sprintf(buf, format4ply, ctr);

    if (ctr ==0)
    {
	int rank;
	MPI_CHECK(MPI_Comm_rank(comm, &rank));

	if(rank == 0)
	    mkdir("ply", S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
    }

    ply_dump(comm, cartcomm, buf, indices, ncells, ntriangles, p, nvertices, false);
}
