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
	int nBytes = n * sizeof(double);
	
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

inline void _copy(double *src, double* dest, int n)
{
	gpuErrchk( cudaMemcpy(dest, src, n * sizeof(double), cudaMemcpyHostToDevice) );
}

////////////////////////////////////////////////////////////////////////////////
/// Short various functions
////////////////////////////////////////////////////////////////////////////////
inline void _fill(double* x, int n, double val)
{
	thrust::fill_n(thrust::device_pointer_cast(x), n, val);
	gpuErrchk( cudaPeekAtLastError() );
}

__global__ void _scal_kernel(double *x, double factor, int n, int start = 0)
{
	const int gid = threadIdx.x + blockDim.x*blockIdx.x + start;
	if (gid >= n) return;
	
	x[gid]  *= factor;
}

inline void _scal(double *x, int n, double factor)
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
__global__ void _K1_kernel(double *y, double *x, int n, double a, int start = 0)
{
	const int gid = threadIdx.x + blockDim.x*blockIdx.x + start;
	if (gid >= n) return;
	
	y[gid] += a * x[gid];
}

inline void _K1(double *y, double *x, int n, double a) // BLAS lev 1: axpy
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
__host__ __device__ inline double _periodize(double val, double l, double l_2)
{
	if (val > l_2)
		return val - l;
	if (val < -l_2)
		return val + l;
	return val;
}

__global__ void _K2_kernel(Particles part, int n, LennardJones potential, double L, double L_2, int start = 0)
{
	const int gid = threadIdx.x + blockDim.x*blockIdx.x + start;
	if (gid >= n) return;
	
	double x = part.x[gid];
	double y = part.y[gid];
	double z = part.z[gid];
	double fx, fy, fz;
	double ax = 0, ay = 0, az = 0;
	
	for (int j = gid+1; j<n; j++)
	{
		double dx = _periodize(part.x[j] - x, L, L_2);
		double dy = _periodize(part.y[j] - y, L, L_2);
		double dz = _periodize(part.z[j] - z, L, L_2);
		
		potential.F(dx, dy, dz,  fx, fy, fz);
		
		ax += fx;
		ay += fy;
		az += fz;
	}
	
	for (int j=0; j<gid; j++)
	{
		double dx = _periodize(part.x[j] - x, L, L_2);
		double dy = _periodize(part.y[j] - y, L, L_2);
		double dz = _periodize(part.z[j] - z, L, L_2);
		
		potential.F(dx, dy, dz,  fx, fy, fz);
		
		ax += fx;
		ay += fy;
		az += fz;
	}
	
	const double m_1 = 1.0 / part.m[gid];
	part.ax[gid] = ax * m_1;
	part.ay[gid] = ay * m_1;
	part.az[gid] = az * m_1;
	
}

void _K2(Particles* part, LennardJones* potential, double L)
{
	double L_2 = 0.5*L;
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
	double xAdd[3];
	
	const int gid = threadIdx.x + blockDim.x*blockIdx.x + start;
	if (gid >= n) return;
	
	double x = part.x[gid];
	double y = part.y[gid];
	double z = part.z[gid];
	
	double fx, fy, fz;
	double ax = 0, ay = 0, az = 0;
	
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
				double dx = part.x[neigh] + xAdd[0] - x;
				double dy = part.y[neigh] + xAdd[1] - y;
				double dz = part.z[neigh] + xAdd[2] - z;
				
				potential.F(dx, dy, dz,
							fx, fy, fz);
				
				ax += fx;
				ay += fy;
				az += fz;
			}
		}
	}
	
	const double m_1 = 1.0 / part.m[gid];
	part.ax[gid] = m_1 * ax;
	part.ay[gid] = m_1 * ay;
	part.az[gid] = m_1 * az;
}

void _K2(Particles* part, LennardJones* potential, Cells<Particles>* cells, double L)
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
__global__ void _normalize_kernel(double *x, int n, double x0, double xmax, int start = 0)
{
	const int gid = threadIdx.x + blockDim.x*blockIdx.x + start;
	if (gid >= n) return;
	
	if (x[gid] > xmax) x[gid] -= xmax - x0;
	if (x[gid] < x0)   x[gid] += xmax - x0;
}

inline void _normalize(double *x, double n, double x0, double xmax)
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
struct Nrg : public unary_function<thrust::tuple<double, double, double, double>, double>
{
	__host__ __device__ double operator() (thrust::tuple<double, double, double, double> val)
	{
		double vx = thrust::get<0>(val);
		double vy = thrust::get<1>(val);
		double vz = thrust::get<2>(val);
		double m  = thrust::get<3>(val);
		
		return m * (vx*vx + vy*vy + vz*vz);
	}
};

double _kineticNrg(Particles* part)
{
	thrust::device_ptr<double> pvx(part->vx);
	thrust::device_ptr<double> pvy(part->vy);
	thrust::device_ptr<double> pvz(part->vz);
	thrust::device_ptr<double> pm (part->m);
	double n = part->n;

	double res = 0.5 * thrust::transform_reduce(make_zip_iterator(make_tuple(pvx,     pvy,     pvz,     pm)),
												make_zip_iterator(make_tuple(pvx + n, pvy + n, pvz + n, pm + n)),
												Nrg(), 0.0, thrust::plus<double>());
	gpuErrchk( cudaPeekAtLastError() );
	return res;
}


////////////////////////////////////////////////////////////////////////////////
/// Potential energy computation
/// No cells
////////////////////////////////////////////////////////////////////////////////
__global__ void _potentialNrg_kernel(Particles part, int n, LennardJones potential, double L, double L_2, int start = 0)
{
	const int gid = threadIdx.x + blockDim.x*blockIdx.x + start;
	if (gid >= n) return;
	
	double x = part.x[gid];
	double y = part.y[gid];
	double z = part.z[gid];
	double res = 0;

	for (int j = gid+1; j<n; j++)
	{
		double dx = _periodize(part.x[j] - x, L, L_2);
		double dy = _periodize(part.y[j] - y, L, L_2);
		double dz = _periodize(part.z[j] - z, L, L_2);
		
		res += potential.V(dx, dy, dz);
	}
	
	for (int j=0; j<gid; j++)
	{
		double dx = _periodize(part.x[j] - x, L, L_2);
		double dy = _periodize(part.y[j] - y, L, L_2);
		double dz = _periodize(part.z[j] - z, L, L_2);
		
		res += potential.V(dx, dy, dz);
	}
	
	part.tmp[gid] = res;
}

double _potentialNrg(Particles* part, LennardJones* potential, double L)
{
	double L_2 = 0.5*L;
	
	int threads, blocks, iters, stride;
	calculateThreadsBlocksIters(threads, blocks, iters, part->n);
	stride = threads*blocks;
	
	for (int i=0; i<iters; i++)
		_potentialNrg_kernel<<<blocks, threads>>>(*part, part->n, *potential, L, L_2, stride * i);
	
	gpuErrchk( cudaPeekAtLastError() );
	
	thrust::device_ptr<double> ptmp(part->tmp);
	double res = 0.5 * thrust::reduce(ptmp, ptmp + part->n, 0.0);
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
	double xAdd[3];
	
	const int gid = threadIdx.x + blockDim.x*blockIdx.x + start;
	if (gid >= n) return;
	
	double x = part.x[gid];
	double y = part.y[gid];
	double z = part.z[gid];
	double res = 0;
	
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
				double dx = part.x[neigh] + xAdd[0] - x;
				double dy = part.y[neigh] + xAdd[1] - y;
				double dz = part.z[neigh] + xAdd[2] - z;
				
				res += potential.V(dx, dy, dz);
			}
		}
	}
	
	part.tmp[gid] = res;
}

double _potentialNrg(Particles* part, LennardJones* potential, Cells<Particles>* cells, double L)
{
	int threads, blocks, iters, stride;
	calculateThreadsBlocksIters(threads, blocks, iters, part->n);
	stride = threads*blocks;
	
	for (int i=0; i<iters; i++)
		_potentialNrg_kernel<<<blocks, threads>>>(*part, part->n, *potential, *cells, stride * i);
	
	gpuErrchk( cudaPeekAtLastError() );
	
	thrust::device_ptr<double> ptmp(part->tmp);
	double res = 0.5 * thrust::reduce(ptmp, ptmp + part->n, 0.0);
	gpuErrchk( cudaPeekAtLastError() );
	return res;
}

////////////////////////////////////////////////////////////////////////////////
/// Linear momentum
////////////////////////////////////////////////////////////////////////////////
struct Momentum : public unary_function<thrust::tuple<double, double>, double>
{
	__host__ __device__ double operator() (thrust::tuple<double, double> val)
	{
		double v = thrust::get<0>(val);
		double m = thrust::get<1>(val);
		
		return m * v;
	}
};


void _linMomentum(Particles* part, double &px, double &py, double &pz)
{
	thrust::device_ptr<double> pvx(part->vx);
	thrust::device_ptr<double> pvy(part->vy);
	thrust::device_ptr<double> pvz(part->vz);
	thrust::device_ptr<double> pm (part->m);
	double n = part->n;
	
	px = thrust::transform_reduce(make_zip_iterator(make_tuple(pvx, pm)),
								  make_zip_iterator(make_tuple(pvx + n, pm + n)),
								  Momentum(), 0.0, thrust::plus<double>());
	gpuErrchk( cudaPeekAtLastError() );
	
	py = thrust::transform_reduce(make_zip_iterator(make_tuple(pvy, pm)),
								  make_zip_iterator(make_tuple(pvy + n, pm + n)),
								  Momentum(), 0.0, thrust::plus<double>());
	gpuErrchk( cudaPeekAtLastError() );
	
	pz = thrust::transform_reduce(make_zip_iterator(make_tuple(pvz, pm)),
								  make_zip_iterator(make_tuple(pvz + n, pm + n)),
								  Momentum(), 0.0, thrust::plus<double>());
	gpuErrchk( cudaPeekAtLastError() );
}


void _angMomentum(Particles* part, double& Lx, double& Ly, double& Lz)
{
	Lx = Ly = Lz = 0;
}


void _centerOfMass(Particles* part, double& mx, double& my, double& mz)
{
	mx = my = mz = 0;
}





