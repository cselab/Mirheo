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
	
	gpuErrchk( cudaMalloc((void**)&part->x,  nBytes) );
	gpuErrchk( cudaMalloc((void**)&part->y,  nBytes) );
	gpuErrchk( cudaMalloc((void**)&part->z,  nBytes) );
	gpuErrchk( cudaMalloc((void**)&part->vx, nBytes) );
	gpuErrchk( cudaMalloc((void**)&part->vy, nBytes) );
	gpuErrchk( cudaMalloc((void**)&part->vz, nBytes) );
	gpuErrchk( cudaMalloc((void**)&part->ax, nBytes) );
	gpuErrchk( cudaMalloc((void**)&part->ay, nBytes) );
	gpuErrchk( cudaMalloc((void**)&part->az, nBytes) );
	gpuErrchk( cudaMalloc((void**)&part->m,  nBytes) );
	
	gpuErrchk( cudaMalloc((void**)&part->tmp,nBytes) );
}

inline void _copy(real *src, real* dest, int n)
{
	gpuErrchk( cudaMemcpy(dest, src, n * sizeof(real), cudaMemcpyHostToDevice) );
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

////////////////////////////////////////////////////////////////////////////////
/// K1
////////////////////////////////////////////////////////////////////////////////
__global__ void _K1_kernel(real *y, real *x, int n, real a, int start = 0)
{
	const int gid = threadIdx.x + blockDim.x*blockIdx.x + start;
	if (gid >= n) return;
	
	y[gid] += a * x[gid];
}

inline void _K1(real *y, real *x, int n, real a) // BLAS lev 1: axpy
{
	int threads, blocks, iters, stride;
	calculateThreadsBlocksIters(threads, blocks, iters, n);
	stride = threads*blocks;
	
	for (int i=0; i<iters; i++)
		_K1_kernel<<<blocks, threads>>>(y, x, n, a, stride * i);
	
	gpuErrchk( cudaPeekAtLastError() );
}


////////////////////////////////////////////////////////////////////////////////
/// K2 for no cells
/// One thread per particle
/// Each interaction is calculated twice
////////////////////////////////////////////////////////////////////////////////
__host__ __device__ inline real _periodize(real val, real l, real l_2)
{
	if (val > l_2)
		return val - l;
	if (val < -l_2)
		return val + l;
	return val;
}

__global__ void _K2_kernel(Particles part, int n, LennardJones potential, real L, real L_2, int start = 0)
{
	const int gid = threadIdx.x + blockDim.x*blockIdx.x + start;
	if (gid >= n) return;
	
	real x = part.x[gid];
	real y = part.y[gid];
	real z = part.z[gid];
	real fx, fy, fz;
	real ax = 0, ay = 0, az = 0;
	
	for (int j = gid+1; j<n; j++)
	{
		real dx = _periodize(part.x[j] - x, L, L_2);
		real dy = _periodize(part.y[j] - y, L, L_2);
		real dz = _periodize(part.z[j] - z, L, L_2);
		
		potential.F(dx, dy, dz,  fx, fy, fz);
		
		ax += fx;
		ay += fy;
		az += fz;
	}
	
	for (int j=0; j<gid; j++)
	{
		real dx = _periodize(part.x[j] - x, L, L_2);
		real dy = _periodize(part.y[j] - y, L, L_2);
		real dz = _periodize(part.z[j] - z, L, L_2);
		
		potential.F(dx, dy, dz,  fx, fy, fz);
		
		ax += fx;
		ay += fy;
		az += fz;
	}
	
	const real m_1 = 1.0 / part.m[gid];
	part.ax[gid] = ax * m_1;
	part.ay[gid] = ay * m_1;
	part.az[gid] = az * m_1;
	
}

void _K2(Particles* part, LennardJones* potential, real L)
{
	real L_2 = 0.5*L;
	int threads, blocks, iters, stride;
	calculateThreadsBlocksIters(threads, blocks, iters, part->n);
	stride = threads*blocks;
	
	for (int i=0; i<iters; i++)
		_K2_kernel<<<blocks, threads>>>(*part, part->n, *potential, L, L_2, stride * i);
	
	gpuErrchk( cudaPeekAtLastError() );
}


////////////////////////////////////////////////////////////////////////////////
/// K2 with cells
/// One thread per particle
/// Each interaction is calculated twice
////////////////////////////////////////////////////////////////////////////////
__global__ void _K2_kernel(Particles part, int n, LennardJones potential, CellsInfo cells, int start = 0)
{
	int ij[3], origIJ[3];
	real xAdd[3];
	
	const int gid = threadIdx.x + blockDim.x*blockIdx.x + start;
	if (gid >= n) return;
	
	real x = part.x[gid];
	real y = part.y[gid];
	real z = part.z[gid];
	
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
				real dx = part.x[neigh] + xAdd[0] - x;
				real dy = part.y[neigh] + xAdd[1] - y;
				real dz = part.z[neigh] + xAdd[2] - z;
				
				potential.F(dx, dy, dz,
							fx, fy, fz);
				
				ax += fx;
				ay += fy;
				az += fz;
			}
		}
	}
	
	const real m_1 = 1.0 / part.m[gid];
	part.ax[gid] = m_1 * ax;
	part.ay[gid] = m_1 * ay;
	part.az[gid] = m_1 * az;
}

void _K2(Particles* part, LennardJones* potential, Cells<Particles>* cells, real L)
{
	int threads, blocks, iters, stride;
	calculateThreadsBlocksIters(threads, blocks, iters, part->n);
	stride = threads*blocks;
	
	for (int i=0; i<iters; i++)
		_K2_kernel<<<blocks, threads>>>(*part, part->n, *potential, *cells, stride * i);
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

////////////////////////////////////////////////////////////////////////////////
/// Kinetic energy computation
////////////////////////////////////////////////////////////////////////////////
struct Nrg : public unary_function<thrust::tuple<real, real, real, real>, real>
{
	__host__ __device__ real operator() (thrust::tuple<real, real, real, real> val)
	{
		real vx = thrust::get<0>(val);
		real vy = thrust::get<1>(val);
		real vz = thrust::get<2>(val);
		real m  = thrust::get<3>(val);
		
		return m * (vx*vx + vy*vy + vz*vz);
	}
};

real _kineticNrg(Particles* part)
{
	thrust::device_ptr<real> pvx(part->vx);
	thrust::device_ptr<real> pvy(part->vy);
	thrust::device_ptr<real> pvz(part->vz);
	thrust::device_ptr<real> pm (part->m);
	real n = part->n;

	real res = 0.5 * thrust::transform_reduce(make_zip_iterator(make_tuple(pvx,     pvy,     pvz,     pm)),
												make_zip_iterator(make_tuple(pvx + n, pvy + n, pvz + n, pm + n)),
												Nrg(), 0.0, thrust::plus<real>());
	gpuErrchk( cudaPeekAtLastError() );
	return res;
}


////////////////////////////////////////////////////////////////////////////////
/// Potential energy computation
/// No cells
////////////////////////////////////////////////////////////////////////////////
__global__ void _potentialNrg_kernel(Particles part, int n, LennardJones potential, real L, real L_2, int start = 0)
{
	const int gid = threadIdx.x + blockDim.x*blockIdx.x + start;
	if (gid >= n) return;
	
	real x = part.x[gid];
	real y = part.y[gid];
	real z = part.z[gid];
	real res = 0;

	for (int j = gid+1; j<n; j++)
	{
		real dx = _periodize(part.x[j] - x, L, L_2);
		real dy = _periodize(part.y[j] - y, L, L_2);
		real dz = _periodize(part.z[j] - z, L, L_2);
		
		res += potential.V(dx, dy, dz);
	}
	
	for (int j=0; j<gid; j++)
	{
		real dx = _periodize(part.x[j] - x, L, L_2);
		real dy = _periodize(part.y[j] - y, L, L_2);
		real dz = _periodize(part.z[j] - z, L, L_2);
		
		res += potential.V(dx, dy, dz);
	}
	
	part.tmp[gid] = res;
}

real _potentialNrg(Particles* part, LennardJones* potential, real L)
{
	real L_2 = 0.5*L;
	
	int threads, blocks, iters, stride;
	calculateThreadsBlocksIters(threads, blocks, iters, part->n);
	stride = threads*blocks;
	
	for (int i=0; i<iters; i++)
		_potentialNrg_kernel<<<blocks, threads>>>(*part, part->n, *potential, L, L_2, stride * i);
	
	gpuErrchk( cudaPeekAtLastError() );
	
	thrust::device_ptr<real> ptmp(part->tmp);
	real res = 0.5 * thrust::reduce(ptmp, ptmp + part->n, 0.0);
	gpuErrchk( cudaPeekAtLastError() );
	return res;
}


////////////////////////////////////////////////////////////////////////////////
/// Potential energy computation
/// With cells
////////////////////////////////////////////////////////////////////////////////
__global__ void _potentialNrg_kernel(Particles part, int n, LennardJones potential, CellsInfo cells, int start = 0)
{
	int ij[3], origIJ[3];
	real xAdd[3];
	
	const int gid = threadIdx.x + blockDim.x*blockIdx.x + start;
	if (gid >= n) return;
	
	real x = part.x[gid];
	real y = part.y[gid];
	real z = part.z[gid];
	real res = 0;
	
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
		
		//printf("%d %d %d \n", ij[0], ij[1], ij[2]);

		// Get our cell id
		cid = cells.getCellIndByIJ(ij);
		int begin = cells.pstart[cid];
		int end   = cells.pstart[cid+1];
		
		// Calculate potential for each particle in our cell
		for (int j=begin; j<end; j++)
		{
			int neigh = cells.pobjids[j];
			if (gid != neigh)
			{
				real dx = part.x[neigh] + xAdd[0] - x;
				real dy = part.y[neigh] + xAdd[1] - y;
				real dz = part.z[neigh] + xAdd[2] - z;
				
				res += potential.V(dx, dy, dz);
			}
		}
	}
	
	part.tmp[gid] = res;
}

real _potentialNrg(Particles* part, LennardJones* potential, Cells<Particles>* cells, real L)
{
	int threads, blocks, iters, stride;
	calculateThreadsBlocksIters(threads, blocks, iters, part->n);
	stride = threads*blocks;
	
	for (int i=0; i<iters; i++)
		_potentialNrg_kernel<<<blocks, threads>>>(*part, part->n, *potential, *cells, stride * i);
	
	gpuErrchk( cudaPeekAtLastError() );
	
	thrust::device_ptr<real> ptmp(part->tmp);
	real res = 0.5 * thrust::reduce(ptmp, ptmp + part->n, 0.0);
	gpuErrchk( cudaPeekAtLastError() );
	return res;
}

////////////////////////////////////////////////////////////////////////////////
/// Linear momentum
////////////////////////////////////////////////////////////////////////////////
struct Momentum : public unary_function<thrust::tuple<real, real>, real>
{
	__host__ __device__ real operator() (thrust::tuple<real, real> val)
	{
		real v = thrust::get<0>(val);
		real m = thrust::get<1>(val);
		
		return m * v;
	}
};


void _linMomentum(Particles* part, real &px, real &py, real &pz)
{
	thrust::device_ptr<real> pvx(part->vx);
	thrust::device_ptr<real> pvy(part->vy);
	thrust::device_ptr<real> pvz(part->vz);
	thrust::device_ptr<real> pm (part->m);
	real n = part->n;
	
	px = thrust::transform_reduce(make_zip_iterator(make_tuple(pvx, pm)),
								  make_zip_iterator(make_tuple(pvx + n, pm + n)),
								  Momentum(), 0.0, thrust::plus<real>());
	gpuErrchk( cudaPeekAtLastError() );
	
	py = thrust::transform_reduce(make_zip_iterator(make_tuple(pvy, pm)),
								  make_zip_iterator(make_tuple(pvy + n, pm + n)),
								  Momentum(), 0.0, thrust::plus<real>());
	gpuErrchk( cudaPeekAtLastError() );
	
	pz = thrust::transform_reduce(make_zip_iterator(make_tuple(pvz, pm)),
								  make_zip_iterator(make_tuple(pvz + n, pm + n)),
								  Momentum(), 0.0, thrust::plus<real>());
	gpuErrchk( cudaPeekAtLastError() );
}


void _angMomentum(Particles* part, real& Lx, real& Ly, real& Lz)
{
	Lx = Ly = Lz = 0;
}


void _centerOfMass(Particles* part, real& mx, real& my, real& mz)
{
	mx = my = mz = 0;
}





