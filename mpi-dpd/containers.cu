#include <sys/stat.h>

#include <rbc-cuda.h>

#include "containers.h"
	    	    
namespace ParticleKernels
{
    __global__ void update_stage1(Particle * p, Acceleration * a, int n, float dt,
				  const float dpdx, const float dpdy, const float dpdz, const bool check = true)
    {
	const float gradp[3] = {dpdx, dpdy, dpdz};
	
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
	    p[pid].u[c] += (a[pid].a[c] - gradp[c]) * dt * 0.5;
    
	for(int c = 0; c < 3; ++c)
	    p[pid].x[c] += p[pid].u[c] * dt;

	if (check)
	    for(int c = 0; c < 3; ++c)
	    {
		assert(p[pid].x[c] >= -L -L/2);
		assert(p[pid].x[c] <= +L +L/2);
	    }
    }

    __global__ void update_stage2_and_1(Particle * p, Acceleration * a, int n, float dt,
					const float dpdx, const float dpdy, const float dpdz, const bool check = true)
    {
	const float gradp[3] = {dpdx, dpdy, dpdz};
	
	assert(blockDim.x * gridDim.x >= n);
    
	const int pid = threadIdx.x + blockDim.x * blockIdx.x;

	if (pid >= n)
	    return;

	for(int c = 0; c < 3; ++c)
	    assert(!isnan(p[pid].u[c]));

	for(int c = 0; c < 3; ++c)
	    assert(!isnan(a[pid].a[c]));

	for(int c = 0; c < 3; ++c)
	{
	    const float mya = a[pid].a[c] - gradp[c];
	    float myu = p[pid].u[c];
	    float myx = p[pid].x[c];

	    myu += mya * dt;
	    myx += myu * dt;
	
	    p[pid].u[c] = myu; 
	    p[pid].x[c] = myx; 
	}

	if (check)
	    for(int c = 0; c < 3; ++c)
	    {
		if (!(p[pid].x[c] >= -L -L/2) || !(p[pid].x[c] <= +L +L/2))
		    printf("Uau: %f %f %f %f %f %f and acc %f %f %f\n", 
			   p[pid].x[0], p[pid].x[1], p[pid].x[2], 
			   p[pid].u[0], p[pid].u[1], p[pid].u[2],
			   a[pid].a[0], a[pid].a[1],a[pid].a[2]);
	    
		assert(p[pid].x[c] >= -L -L/2);
		assert(p[pid].x[c] <= +L +L/2);
	    }
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
}

void ParticleArray::update_stage1(const float gradpressure[3])
{
    if (size)
	ParticleKernels::update_stage1<<<(xyzuvw.size + 127) / 128, 128 >>>(
	    xyzuvw.data, axayaz.data, xyzuvw.size, dt,
	    gradpressure[0], gradpressure[1], gradpressure[2] , false);
}

void  ParticleArray::update_stage2_and_1(const float gradpressure[3])
{
    if (size)
	ParticleKernels::update_stage2_and_1<<<(xyzuvw.size + 127) / 128, 128 >>>
	    (xyzuvw.data, axayaz.data, xyzuvw.size, dt,
	     gradpressure[0], gradpressure[1], gradpressure[2], false);
}

void ParticleArray::resize(int n)
{
    size = n;
    
    xyzuvw.resize(n);
    axayaz.resize(n);
    
    CUDA_CHECK(cudaMemset(axayaz.data, 0, sizeof(Acceleration) * size));
}

void ParticleArray::clear_velocity()
{
    if (size)
	ParticleKernels::clear_velocity<<<(xyzuvw.size + 127) / 128, 128 >>>(xyzuvw.data, xyzuvw.size);
}

void CollectionRBC::resize(const int count)
{
    nrbcs = count;

    ParticleArray::resize(count * nvertices);
}
    
struct TransformedExtent
{
    float com[3];
    float transform[4][4];
};

CollectionRBC::CollectionRBC(MPI_Comm cartcomm, const int L, const string path2ic): 
    cartcomm(cartcomm), L(L), nrbcs(0), path2xyz("rbcs.xyz"), format4ply("ply/rbcs-%04d.ply"), path2ic("rbcs-ic.txt"), dumpcounter(0)
{
    MPI_CHECK(MPI_Comm_rank(cartcomm, &myrank));
    MPI_CHECK( MPI_Cart_get(cartcomm, 3, dims, periods, coords) );
    
    CudaRBC::Extent extent;
    CudaRBC::setup(nvertices, extent);

    assert(extent.xmax - extent.xmin < L);
    assert(extent.ymax - extent.ymin < L);
    assert(extent.zmax - extent.zmin < L);

    CudaRBC::get_triangle_indexing(indices, ntriangles);
}

void CollectionRBC::setup()
{
    vector<TransformedExtent> allrbcs;

    if (myrank == 0)
    {
	//read transformed extent from file
	FILE * f = fopen(path2ic.c_str(), "r");
	printf("READING FROM: <%s>\n", path2ic.c_str());
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
	printf("Instantiating %d CELLs from...<%s>\n", (int)allrbcs.size(), path2ic.c_str());

    int allrbcs_count = allrbcs.size();
    MPI_CHECK(MPI_Bcast(&allrbcs_count, 1, MPI_INT, 0, cartcomm));

    allrbcs.resize(allrbcs_count);
    
    const int nfloats_per_entry = sizeof(TransformedExtent) / sizeof(float);
    assert( sizeof(TransformedExtent) % sizeof(float) == 0);

    MPI_CHECK(MPI_Bcast(&allrbcs.front(), nfloats_per_entry * allrbcs_count, MPI_FLOAT, 0, cartcomm));

    vector<TransformedExtent> good;

    for(vector<TransformedExtent>::iterator it = allrbcs.begin(); it != allrbcs.end(); ++it)
    {
	bool inside = true;

	for(int c = 0; c < 3; ++c)
	    inside &= it->com[c] >= coords[c] * L && it->com[c] < (coords[c] + 1) * L;

	if (inside)
	{
	    for(int c = 0; c < 3; ++c)
		it->transform[c][3] -= (coords[c] + 0.5) * L;

	    good.push_back(*it);
	}
    }
    
    resize(good.size());

    for(int i = 0; i < good.size(); ++i)
	_initialize((float *)(xyzuvw.data + nvertices * i), good[i].transform);
	//CudaRBC::initialize((float *)(xyzuvw.data + nvertices * i), good[i].transform);
}

void CollectionRBC::_initialize(float *device_xyzuvw, const float (*transform)[4])
{
    CudaRBC::initialize(device_xyzuvw, transform);
}

void CollectionRBC::update_stage1()
{
    const float dpdx[3] = {0, 0, 0};
    ParticleArray::update_stage1(dpdx);
}

void CollectionRBC::update_stage2_and_1()
{
    const float dpdx[3] = {0, 0, 0};
    ParticleArray::update_stage2_and_1(dpdx);
}

void CollectionRBC::remove(const int * const entries, const int nentries)
{
    std::vector<bool > marks(nrbcs, true);

    for(int i = 0; i < nentries; ++i)
	marks[entries[i]] = false;

    std::vector< int > survivors;
    for(int i = 0; i < nrbcs; ++i)
	if (marks[i])
	    survivors.push_back(i);

    const int nsurvived = survivors.size();

    SimpleDeviceBuffer<Particle> survived(nvertices * nsurvived);

    for(int i = 0; i < nsurvived; ++i)
	CUDA_CHECK(cudaMemcpy(survived.data + nvertices * i, data() + nvertices * survivors[i], 
			      sizeof(Particle) * nvertices, cudaMemcpyDeviceToDevice));
	    
    resize(nsurvived);

    CUDA_CHECK(cudaMemcpy(xyzuvw.data, survived.data, sizeof(Particle) * survived.size, cudaMemcpyDeviceToDevice));
}

void CollectionRBC::dump(MPI_Comm comm)
{
    int& ctr = dumpcounter;
    const bool firsttime = ctr == 0;
	    
    const int n = size;

    Particle * p = new Particle[n];
    Acceleration * a = new Acceleration[n];

    CUDA_CHECK(cudaMemcpy(p, xyzuvw.data, sizeof(Particle) * n, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(a, axayaz.data, sizeof(Acceleration) * n, cudaMemcpyDeviceToHost));
		   
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
	xyz_dump(comm, path2xyz.c_str(), "cell-particles", p, n,  L, !firsttime);

    
    

    char buf[200];
    sprintf(buf, format4ply.c_str(), ctr);

    if (ctr ==0)
    {
	int rank;
	MPI_CHECK(MPI_Comm_rank(comm, &rank));
		
	if(rank == 0)
	    mkdir("ply", S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
    }
	    
    ply_dump(comm, buf, indices, nrbcs, ntriangles, p, nvertices, L, false);
		    
    delete [] p;
    delete [] a;

    ++ctr;
}
