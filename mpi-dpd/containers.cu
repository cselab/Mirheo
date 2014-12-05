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

struct TransformedExtent
{
    float transform[4][4];

    float xmin[3], xmax[3],local_xmin[3], local_xmax[3];
       
    void build_transform(CudaRBC::Extent extent, const int L)
	{
	    const float angles[3] = { 
		0.25 * (drand48() - 0.5) * 2 * M_PI, 
		M_PI * 0.5 + 0.25 * (drand48() * 2 - 1) * M_PI,
		0.25 * (drand48() - 0.5) * 2 * M_PI
	    };

	    for(int i = 0; i < 4; ++i)
		for(int j = 0; j < 4; ++j)
		    transform[i][j] = i == j;

	    for(int i = 0; i < 3; ++i)
		transform[i][3] = - 0.5 * (local_xmin[i] + local_xmax[i]);

	    for(int d = 0; d < 3; ++d)
	    {
		const float c = cos(angles[d]);
		const float s = sin(angles[d]);

		float tmp[4][4];

		for(int i = 0; i < 4; ++i)
		    for(int j = 0; j < 4; ++j)
			tmp[i][j] = i == j;

		if (d == 0)
		{
		    tmp[0][0] = tmp[1][1] = c;
		    tmp[0][1] = -(tmp[1][0] = s);
		} 
		else 
		    if (d == 1)
		    {
			tmp[0][0] = tmp[2][2] = c;
			tmp[0][2] = -(tmp[2][0] = s);
		    }
		    else
		    {  
			tmp[1][1] = tmp[2][2] = c;
			tmp[1][2] = -(tmp[2][1] = s);
		    }

		float res[4][4];
		for(int i = 0; i < 4; ++i)
		    for(int j = 0; j < 4; ++j)
		    {
			float s = 0;
			    
			for(int k = 0; k < 4; ++k)
			    s += transform[i][k] * tmp[k][j];

			res[i][j] = s;
		    }

		for(int i = 0; i < 4; ++i)
		    for(int j = 0; j < 4; ++j)
			transform[i][j] = res[i][j];
	    }

	    for(int i = 0; i < 3; ++i)
		transform[i][3] += (drand48() - 0.5) * (L - 4);
	}

    void apply(float x[3], float y[3])
	{
	    for(int i = 0; i < 3; ++i)
		y[i] = transform[i][0] * x[0] + transform[i][1] * x[1] + transform[i][2] * x[2] + transform[i][3];
	}

    TransformedExtent(CudaRBC::Extent extent, const int L)
	{
	    local_xmin[0] = extent.xmin;
	    local_xmin[1] = extent.ymin;
	    local_xmin[2] = extent.zmin;
		
	    local_xmax[0] = extent.xmax;
	    local_xmax[1] = extent.ymax;
	    local_xmax[2] = extent.zmax;
	
	    build_transform(extent, L);

	    for(int i = 0; i < 8; ++i)
	    {
		const int idx[3] = { i % 2, (i/2) % 2, (i/4) % 2 };

		float local[3];
		for(int c = 0; c < 3; ++c)
		    local[c] = idx[c] ? local_xmax[c] : local_xmin[c];

		float world[3];

		apply(local, world);

		if (i == 0)
		    for(int c = 0; c < 3; ++c)
			xmin[c] = xmax[c] = world[c];
		else
		    for(int c = 0; c < 3; ++c)
		    {
			xmin[c] = min(xmin[c], world[c]);
			xmax[c] = max(xmax[c], world[c]);
		    }
	    }
	}

    bool collides(const TransformedExtent a, const  float tol)
	{
	    int s[3], e[3];
	    for(int c = 0; c < 3; ++c)
	    {
		s[c] = max(xmin[c], a.xmin[c]);
		e[c] = min(xmax[c], a.xmax[c]);

		if (s[c] -e[c] >= tol)
		    return false;
	    }

	    return true;
	}
};


void CollectionRBC::resize(const int count)
{
    nrbcs = count;

    ParticleArray::resize(count * nvertices);
}

    
CollectionRBC::CollectionRBC(const int L): L(L), nrbcs(0)
{
    CudaRBC::Extent extent;
    CudaRBC::setup(nvertices, extent);

    /*
    printf("extent: %f %f %f %f %f %f\n",
	   extent.xmax , extent.xmin,
	   extent.ymax , extent.ymin,
	   extent.zmax , extent.zmin);	   
    */
    
    assert(extent.xmax - extent.xmin < L);
    assert(extent.ymax - extent.ymin < L);
    assert(extent.zmax - extent.zmin < L);

    vector<TransformedExtent*> good;
	    
    for(int attempt = 0; attempt < 1000; ++attempt)
    {
	TransformedExtent* t = new TransformedExtent(extent, L);
		
	bool noncolliding = true;
	for(int i = 0; i < good.size() && noncolliding; ++i)
	    noncolliding &= !t->collides(*good[i], 0.00);
		
	if (noncolliding)
	    good.push_back(t);
    }
	    
    resize(good.size());

    for(int i = 0; i < good.size(); ++i)
	CudaRBC::initialize((float *)(xyzuvw.data + nvertices * i), good[i]->transform);
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
    static int ctr = 0;
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
	xyz_dump(comm, "rbcs.xyz", "rbcparticles", p, n,  L, !firsttime);

    int (*indices)[3];
    int ntriangles;
    CudaRBC::get_triangle_indexing(indices, ntriangles);

    char buf[200];
    sprintf(buf, "ply/rbcs-%04d.ply", ctr);

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
