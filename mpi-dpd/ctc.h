#pragma once

#include "redistribute-rbcs.h"
#include "rbc-interactions.h"

#include <ctc-cuda.h>

class RedistributeCTCs : public RedistributeRBCs
{
    void _compute_extents(const Particle * const xyzuvw, const int nrbcs)
    {
	assert(sizeof(CudaCTC::Extent) == sizeof(CudaRBC::Extent));

	for(int i = 0; i < nrbcs; ++i)
	    CudaCTC::extent_nohost(stream, (float *)(xyzuvw + nvertices * i), (CudaCTC::Extent *)(extents.devptr + i));
    }

public:

RedistributeCTCs(MPI_Comm _cartcomm, const int L):RedistributeRBCs(_cartcomm, L)
    {
	nvertices = CudaCTC::get_nvertices();
    }
};
  
class ComputeInteractionsCTC : public ComputeInteractionsRBC
{
    void _compute_extents(const Particle * const xyzuvw, const int nrbcs)
    {
	assert(sizeof(CudaCTC::Extent) == sizeof(CudaRBC::Extent));

	for(int i = 0; i < nrbcs; ++i)
	    CudaCTC::extent_nohost(stream, (float *)(xyzuvw + nvertices * i), (CudaCTC::Extent *)(extents.devptr + i));
    }

    void _internal_forces(const Particle * const rbcs, const int nrbcs, Acceleration * accrbc)
    {
	for(int i = 0; i < nrbcs; ++i)
	    CudaCTC::forces_nohost(stream, (float *)(rbcs + nvertices * i), (float *)(accrbc + nvertices * i));
    }

public:

ComputeInteractionsCTC(MPI_Comm _cartcomm, int L): ComputeInteractionsRBC(_cartcomm, L)
    {
	local_trunk = Logistic::KISS(598 - myrank, 20383 + myrank, 129037, 2580);

	nvertices = CudaCTC::get_nvertices();
    }
};

class CollectionCTC : public CollectionRBC
{
    void _initialize(float *device_xyzuvw, const float (*transform)[4])
    {
	CudaCTC::initialize(device_xyzuvw, transform);
    }

public:

CollectionCTC(MPI_Comm cartcomm, const int L):CollectionRBC(cartcomm, L)
    {
	

	CudaCTC::Extent extent;
	CudaCTC::setup(nvertices, extent, dt);

	assert(extent.xmax - extent.xmin < L);
	assert(extent.ymax - extent.ymin < L);
	assert(extent.zmax - extent.zmin < L);

	CudaCTC::get_triangle_indexing(indices, ntriangles);

	path2xyz = "ctcs.xyz";
	format4ply = "ply/ctcs-%04d.ply";
	path2ic = "ctcs-ic.txt";
    }
};
