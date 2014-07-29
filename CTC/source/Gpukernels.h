/*
 *  Gpukernels.h
 *  hpchw
 *
 *  Created by Dmitry Alexeev on 23.10.13.
 *  Copyright 2013 ETH Zurich. All rights reserved.
 *
 */

#include "CellList.h"
#include "Potential.h"

#include "thrust/device_ptr.h"
#include "thrust/device_vector.h"
#include "thrust/transform.h"
#include "thrust/transform_reduce.h"
#include "thrust/reduce.h"

#include "Misc.h"

// Error checking on gpu
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, char *file, int line, bool abort=true)
{
	if (code != cudaSuccess) 
	{
		fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

inline int nofMPs()
{
	int device;
	cudaDeviceProp prop;
	gpuErrchk( cudaGetDevice(&device) );
	gpuErrchk( cudaGetDeviceProperties(&prop, device) );
	return prop.multiProcessorCount;
}

inline void calculateThreadsBlocksIters(int& threads, int& blocks, int& iters, int n)
{
	static const int maxblocks = 10 * nofMPs();
	
	threads = (n >= maxblocks * 128) ? 256 : 128;
	blocks  = (n + threads - 1) / threads;
	iters   = blocks / maxblocks + 1;
	blocks  = (iters > 1) ? maxblocks : blocks;
}

////////////////////////////////////////////////////////////////////////////////
/// Allocations for GPU
////////////////////////////////////////////////////////////////////////////////
inline void _allocate(Particles* part)
{
	int n = part->n;
	int nBytes = n * sizeof(real);
	
	gpuErrchk( cudaMalloc((void**)&part->xdata,  3*nBytes) );
	gpuErrchk( cudaMalloc((void**)&part->vdata,  3*nBytes) );
	gpuErrchk( cudaMalloc((void**)&part->adata,  3*nBytes) );
	gpuErrchk( cudaMalloc((void**)&part->m,      nBytes) );
	
	//gpuErrchk( cudaMalloc((void**)&part->tmp,nBytes) );
}

inline void _copy2device(Particles *src, Particles* dest)
{
	gpuErrchk( cudaMemcpy(dest->xdata, src->xdata, 3 * src->n * sizeof(real), cudaMemcpyHostToDevice) );
	gpuErrchk( cudaMemcpy(dest->vdata, src->vdata, 3 * src->n * sizeof(real), cudaMemcpyHostToDevice) );
}

inline void _copy2host(Particles *src, Particles* dest)
{
//    static real* xtmp, vtmp;
//    static bool allocated = false;
//    
//    if (!allocated)
//    {
//        gpuErrchk( cudaMalloc((void**)&xtmp,  3 * src->n * sizeof(real)) );
//        gpuErrchk( cudaMalloc((void**)&vtmp,  3 * src->n * sizeof(real)) );
//        allocated = true;
//    }
//    
//    gpuErrchk( cudaMemcpy(src->xdata, xtmp, 3 * src->n * sizeof(real), cudaMemcpyHostToHost) );
//    gpuErrchk( cudaMemcpy(src->vdata, vtmp, 3 * src->n * sizeof(real), cudaMemcpyHostToHost) );
    
	gpuErrchk( cudaMemcpy(dest->xdata, src->xdata, 3 * src->n * sizeof(real), cudaMemcpyDeviceToHost) );
	gpuErrchk( cudaMemcpy(dest->vdata, src->vdata, 3 * src->n * sizeof(real), cudaMemcpyDeviceToHost) );
}


////////////////////////////////////////////////////////////////////////////////
/// Short various functions
////////////////////////////////////////////////////////////////////////////////
inline void _fill(real* x, int n, real val)
{
	thrust::fill_n(thrust::device_pointer_cast(x), n, val);
	gpuErrchk( cudaPeekAtLastError() );
}

__global__ void _scal_kernel(real *x, real factor, int n, int start = 0)
{
	const int gid = threadIdx.x + blockDim.x*blockIdx.x + start;
	if (gid >= n) return;
	
	x[gid]  *= factor;
}

inline void _scal(real *x, int n, real factor)
{
	int threads, blocks, iters, stride;
	calculateThreadsBlocksIters(threads, blocks, iters, n);
	stride = threads*blocks;
	
	for (int i=0; i<iters; i++)
		_scal_kernel<<<blocks, threads>>>(x, factor, n, stride * i);
	gpuErrchk( cudaPeekAtLastError() );
}


__global__ void _nuscal_kernel(real *y, real *x, real factor, int n, int start = 0)
{
	const int gid = threadIdx.x + blockDim.x*blockIdx.x + start;
	if (gid >= n) return;
	
	y[gid] /= x[gid/3];
}

inline void _nuscal(real *y, real *x, int n, real factor)
{
	int threads, blocks, iters, stride;
	calculateThreadsBlocksIters(threads, blocks, iters, n);
	stride = threads*blocks;
	
	for (int i=0; i<iters; i++)
		_nuscal_kernel<<<blocks, threads>>>(x, y, factor, n, stride * i);
	gpuErrchk( cudaPeekAtLastError() );
}
////////////////////////////////////////////////////////////////////////////////
/// K1
////////////////////////////////////////////////////////////////////////////////
__global__ void _axpy_kernel(real *y, real *x, int n, real a, int start = 0)
{
	const int gid = threadIdx.x + blockDim.x*blockIdx.x + start;
	if (gid >= n) return;
	
	y[gid] += a * x[gid];
}

inline void _axpy(real *y, real *x, int n, real a) // BLAS lev 1: axpy
{
	int threads, blocks, iters, stride;
	calculateThreadsBlocksIters(threads, blocks, iters, n);
	stride = threads*blocks;
	
	for (int i=0; i<iters; i++)
		_axpy_kernel<<<blocks, threads>>>(y, x, n, a, stride * i);
	
	gpuErrchk( cudaPeekAtLastError() );
}


////////////////////////////////////////////////////////////////////////////////
/// K2 with cells
/// One thread per particle
/// Each interaction is calculated twice
////////////////////////////////////////////////////////////////////////////////
template<typename Interactor>
__global__ void _gpuForces_kernel(Particles part, CellsInfo cells, int n, Interactor inter, int t)
{
	int ij[3], origIJ[3];
	real xAdd[3];
	
	const int gid = threadIdx.x + blockDim.x*blockIdx.x + start;
	if (gid >= n) return;
	
	real x = part.x(gid);
	real y = part.y(gid);
	real z = part.z(gid);
	
	real fx, fy, fz;
	real ax = 0, ay = 0, az = 0;
	
	// Get id of the cell where the particle is situated and its coordinates
	int cid = cells.which(x, y, z);
	cells.getCellIJByInd(cid, origIJ);
	
	// Loop over all neigboring cells
	for (int direction_code = 0; direction_code < 27; direction_code++)
	{
		// Get the coordinates of the cell we work on
		ij[0] = (direction_code % 3) - 1;
		ij[1] = (direction_code/3 % 3) - 1;
		ij[2] = (direction_code/9 % 3) - 1;
		
		// Resolve periodicity
		for (int k=0; k<3; k++)
			ij[k] += origIJ[k];

		cells.correct(ij, xAdd);
		
		// Get our cell id
		cid = cells.getCellIndByIJ(ij);
		int begin = cells.pstart[cid];
		int end   = cells.pstart[cid+1];
		
		// Calculate forces for all the particles in our cell		
		for (int j=begin; j<end; j++)
		{
			int neigh = cells.pobjids[j];
			if (gid != neigh)
			{
				real dx = part.x(neigh) + xAdd[0] - x;
				real dy = part.y(neigh) + xAdd[1] - y;
				real dz = part.z(neigh) + xAdd[2] - z;
                
                const real r2 = dx*dx + dy*dy + dz*dz;
                
                real vx = part.vx(dst) - part.vx(src);
                real vy = part.vy(dst) - part.vy(src);
                real vz = part.vz(dst) - part.vz(src);
				
				inter.F(dx, dy, dz,  vx, vy, vz,  r2,  fx, fy, fz,  src, dst, t);
				
				ax += fx;
				ay += fy;
				az += fz;
			}
		}
	}
	
	const real m_1 = 1.0 / part.m[gid];
	part.ax(gid) = m_1 * ax;
	part.ay(gid) = m_1 * ay;
	part.az(gid) = m_1 * az;
}

template<typename Interactor>
void computeForces(Particles** part, Cells<Particles>*** cells, int a, int b, Interactor &inter, int time)
{
    if (a != b) return;
    
	int threads, blocks, iters, stride;
	calculateThreadsBlocksIters(threads, blocks, iters, part->n);
	stride = threads*blocks;
	
    _gpuForces_kernel<<<blocks, threads>>>(*part[a], *cells[a][a], part[a]->n, inter);
    
	gpuErrchk( cudaPeekAtLastError() );
}

////////////////////////////////////////////////////////////////////////////////
/// Normalization kernel. Maintains periodicity
////////////////////////////////////////////////////////////////////////////////
__global__ void _normalize_kernel(real *x, int n, real x0, real xmax, int start = 0)
{
	const int gid = threadIdx.x + blockDim.x*blockIdx.x + start;
	if (gid >= n) return;
	
	if (x[gid] > xmax) x[gid] -= xmax - x0;
	if (x[gid] < x0)   x[gid] += xmax - x0;
}

inline void _normalize(real *x, real n, real x0, real xmax)
{
	int threads, blocks, iters, stride;
	calculateThreadsBlocksIters(threads, blocks, iters, n);
	stride = threads*blocks;
	
	for (int i=0; i<iters; i++)
		_normalize_kernel<<<blocks, threads>>>(x, n, x0, xmax, stride * i);
	
	gpuErrchk( cudaPeekAtLastError() );
}



