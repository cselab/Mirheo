/*
 *  CellList3DVector.h
 *  hpchw
 *
 *  Created by Dmitry Alexeev on 17.10.13.
 *  Copyright 2013 ETH Zurich. All rights reserved.
 *
 *  Cell list as it should be used with CUDA as well
 *  Check http://http.developer.nvidia.com/GPUGems3/gpugems3_ch32.html
 */

#pragma once

#include <cmath>
#include <iostream>

#include "thrust/sequence.h"
#include "thrust/sort.h"
#include "thrust/device_ptr.h"
#include "thrust/device_vector.h"
#include "thrust/binary_search.h"

#include "Misc.h"

#define Dims 3

struct GenerateKeys;

class CellsInfo
{
public:
	int n0, n1, n2;
	int nTot;
	
	real xLow0, xLow1, xLow2, xHigh0, xHigh1, xHigh2;
	real h0, h1, h2;
	real l0, l1, l2;
	int mult0, mult1, mult2;

	int* pcellids;
	int* pobjids;
	int* pstart;
	
public:
	
	inline CellsInfo(real, real[Dims], real[Dims]);
	__host__ __device__ inline int  getCellIndByIJ(int[Dims]);
	__host__ __device__ inline void getCellIJByInd(int, int[Dims]);
	__host__ __device__ inline int  which(real x0, real x1, real x2);
	__host__ __device__ inline void correct(int *ij, real *xAdd);
};	

// Cells for objects described as Structure of Arrays in Objects
template <typename Objects>
class Cells : public CellsInfo
{
    real *xhost;
    real *yhost;
    real *zhost;
    
public:
	Objects* objects;
	int      nObj;
	thrust::device_vector<int>	cellids;	// To which cell each object belongs
	thrust::device_vector<int>	objids;		// A mapping between spatially sorted and original objects
	thrust::device_vector<int>	start;		// Index, at which contents of a cell starts in objids array
	
	 // A functor computing the id of the cell to which the object belongs to
	GenerateKeys* genKeys;
	
public:
	Cells(Objects*, int, real, real[Dims], real[Dims]);
	
	void migrate();
};

struct GenerateKeys
{
	CellsInfo cells;
	
	GenerateKeys(CellsInfo cells) : cells(cells) {};
	__host__ __device__ int operator()(thrust::tuple<real, real, real> coo)
	{
		return cells.which(thrust::get<0>(coo),
				           thrust::get<1>(coo),
				           thrust::get<2>(coo));
	}
};

inline CellsInfo::CellsInfo(real size, real low[Dims], real high[Dims])
{
	nTot = 1;

	// 0
	xLow0  = low[0];
	xHigh0 = high[0];
	l0     = xHigh0 - xLow0;
	n0     = l0 / size;

	if (n0 == 0) n0 = 1;
	h0 = l0 / n0;

	nTot *= n0;

	// 1
	xLow1  = low[1];
	xHigh1 = high[1];
	l1     = xHigh1 - xLow1;
	n1     = l1 / size;

	if (n1 == 0) n1 = 1;
	h1 = l1 / n1;

	nTot *= n1;

	// 2
	xLow2  = low[2];
	xHigh2 = high[2];
	l2     = xHigh2 - xLow2;
	n2     = l2 / size;

	if (n2 == 0) n2 = 1;
	h2 = l2 / n2;

	nTot *= n2;
	
	mult2 = 1;
	mult1 = mult2 * n1;
	mult0 = mult1 * n0;
}

__host__ __device__ inline int CellsInfo::getCellIndByIJ(int ij[Dims])
{
	int res = mult0*ij[0] + mult1*ij[1] + mult2*ij[2];
	
	return res;
}

__host__ __device__ inline void CellsInfo::getCellIJByInd(int id, int ij[Dims])
{
	ij[0] = id / mult0;
	id = id % mult0;

	ij[1] = id / mult1;
	id = id % mult1;

	ij[2] = id / mult2;
	id = id % mult2;
}

__host__ __device__ inline int CellsInfo::which(real x0, real x1, real x2)
{
	int ij[Dims];
	
	ij[0] = floor((x0 - xLow0) / h0);
	ij[1] = floor((x1 - xLow1) / h1);
	ij[2] = floor((x2 - xLow2) / h2);

	if (ij[0] >= n0) ij[0] = n0-1;
	if (ij[0] <  0)  ij[0] = 0;
	
	if (ij[1] >= n1) ij[1] = n1-1;
	if (ij[1] <  0)  ij[1] = 0;
	
	if (ij[2] >= n2) ij[2] = n2-1;
	if (ij[2] <  0)  ij[2] = 0;

	return getCellIndByIJ(ij);
}

// Modify cell coordinates and compute real coordinate corrections
__host__ __device__ inline void CellsInfo::correct(int *ij, real *xAdd)
{
	xAdd[0] = xAdd[1] = xAdd[2] = 0;
	// 0
	if (ij[0] < 0)
	{
		xAdd[0] = -l0;
		ij[0]   =  n0-1;
	}

	if (ij[0] >= n0)
	{
		xAdd[0] = +l0;
		ij[0]   =  0;
	}

	// 1
	if (ij[1] < 0)
	{
		xAdd[1] = -l1;
		ij[1]   =  n1-1;
	}

	if (ij[1] >= n1)
	{
		xAdd[1] = +l1;
		ij[1]   =  0;
	}

	// 2
	if (ij[2] < 0)
	{
		xAdd[2] = -l2;
		ij[2]   =  n2-1;
	}

	if (ij[2] >= n2)
	{
		xAdd[2] = +l2;
		ij[2]   =  0;
	}
}

template <typename Object>
Cells<Object>::Cells(Object* obj, int nObj, real size, real low[Dims], real high[Dims]):
objects(obj), nObj(nObj), CellsInfo(size, low, high)
{
	start.resize(this->nTot + 1);
	objids.resize(this->nObj);
	cellids.resize(this->nObj);
	
	this->pstart   = raw_pointer_cast(&start[0]);
	this->pobjids  = raw_pointer_cast(&objids[0]);
	this->pcellids = raw_pointer_cast(&cellids[0]);
    
    xhost = new real[this->nObj];
    yhost = new real[this->nObj];
    zhost = new real[this->nObj];
	
	genKeys = new GenerateKeys(*this);
	
	migrate();
}

template <typename Object>
void Cells<Object>::migrate()
{
    for (int i=0; i<this->nObj; i++)
    {
        xhost[i] = this->objects->xdata[3*i+0];
        yhost[i] = this->objects->xdata[3*i+1];
        zhost[i] = this->objects->xdata[3*i+2];
    }
	// Wrap the arrays with thrust vectors
	thrust::device_ptr<real> xptr = thrust::device_pointer_cast(xhost);
	thrust::device_ptr<real> yptr = thrust::device_pointer_cast(yhost);
	thrust::device_ptr<real> zptr = thrust::device_pointer_cast(zhost);
	
	// Calculate index of a cell where a particle is situated
	thrust::transform(make_zip_iterator(make_tuple(xptr, yptr, zptr)),
					  make_zip_iterator(make_tuple(xptr + this->nObj, yptr + this->nObj, zptr + this->nObj)),
					  cellids.begin(), *genKeys);	
	
	// Rearrange in such a way, that particles in the same cell are close
	thrust::sequence(objids.begin(), objids.end(), 0);		
	thrust::sort_by_key(cellids.begin(), cellids.end(), objids.begin());
	
	
	thrust::lower_bound(cellids.begin(), cellids.end(), 
						thrust::make_counting_iterator(0), thrust::make_counting_iterator(this->nTot + 1),
						start.begin());
}





