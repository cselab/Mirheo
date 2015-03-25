/*
 *  ctc.h
 *  Part of CTC/mpi-dpd/
 *
 *  Created and authored by Diego Rossinelli on 2014-12-18.
 *  Copyright 2015. All rights reserved.
 *
 *  Users are NOT authorized
 *  to employ the present software for their own publications
 *  before getting a written permission from the author of this file.
 */

#pragma once

#include "redistribute-rbcs.h"
#include "rbc-interactions.h"
#include "minmax-massimo.h"

#include <ctc-cuda.h>

class RedistributeCTCs : public RedistributeRBCs
{
    void _compute_extents(const Particle * const xyzuvw, const int nrbcs, cudaStream_t stream)
    {
	assert(sizeof(CudaCTC::Extent) == sizeof(CudaRBC::Extent));
#if 1
	minmax_massimo(xyzuvw, nvertices, nrbcs, minextents.devptr, maxextents.devptr, stream);
#else
	for(int i = 0; i < nrbcs; ++i)
	    CudaCTC::extent_nohost(stream, (float *)(xyzuvw + nvertices * i), (CudaCTC::Extent *)(extents.devptr + i));
#endif
    }

public:

RedistributeCTCs(MPI_Comm _cartcomm):RedistributeRBCs(_cartcomm)
    {
	if (ctcs)
	{
	    CudaCTC::Extent host_extent;
	    CudaCTC::setup(nvertices, host_extent, dt);
	}
    }
};
  
class ComputeInteractionsCTC : public ComputeInteractionsRBC
{
    void _compute_extents(const Particle * const xyzuvw, const int nrbcs, cudaStream_t stream)
    {
	assert(sizeof(CudaCTC::Extent) == sizeof(CudaRBC::Extent));
#if 1
	minmax_massimo(xyzuvw, nvertices, nrbcs, minextents.devptr, maxextents.devptr, stream);
#else
	for(int i = 0; i < nrbcs; ++i)
	    CudaCTC::extent_nohost(stream, (float *)(xyzuvw + nvertices * i), (CudaCTC::Extent *)(extents.devptr + i));
#endif
    }

    void _internal_forces(const Particle * const rbcs, const int nrbcs, Acceleration * accrbc, cudaStream_t stream)
    {
	for(int i = 0; i < nrbcs; ++i)
	    CudaCTC::forces_nohost(stream, (float *)(rbcs + nvertices * i), (float *)(accrbc + nvertices * i));
    } 

public:

ComputeInteractionsCTC(MPI_Comm _cartcomm) : ComputeInteractionsRBC(_cartcomm)
    {
	local_trunk = Logistic::KISS(598 - myrank, 20383 + myrank, 129037, 2580);

	if (ctcs)
	{
	    CudaCTC::Extent host_extent;
	    CudaCTC::setup(nvertices, host_extent, dt);
	}
    }
};

class CollectionCTC : public CollectionRBC
{
    void _initialize(float *device_xyzuvw, const float (*transform)[4])
    {
	CudaCTC::initialize(device_xyzuvw, transform);
    }

public:

CollectionCTC(MPI_Comm cartcomm) : CollectionRBC(cartcomm)
    {
	if (ctcs)
	{
	    CudaCTC::Extent extent;
	    CudaCTC::setup(nvertices, extent, dt);

	    assert(extent.xmax - extent.xmin < XSIZE_SUBDOMAIN);
	    assert(extent.ymax - extent.ymin < YSIZE_SUBDOMAIN);
	    assert(extent.zmax - extent.zmin < ZSIZE_SUBDOMAIN);
	    
	    CudaCTC::get_triangle_indexing(indices, ntriangles);
	}
	
	path2xyz = "ctcs.xyz";
	format4ply = "ply/ctcs-%04d.ply";
	path2ic = "ctcs-ic.txt";
    }
};
