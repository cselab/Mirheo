#pragma once

#include <utility>

#include <cuda_runtime.h>

void build_clists(float * const device_xyzuvw, int np, const float rc, 
		  const int xcells, const int ycells, const int zcells,
		  const float xdomainstart, const float ydomainstart, const float zdomainstart,
		  int * const host_order, int * device_cellsstart, int * device_cellscount,
		  std::pair<int, int *> * nonemptycells, cudaStream_t stream);

void forces_dpd_cuda_nohost(const float * const _xyzuvw, float * const _axayaz,  const int np,
			    const int * const cellsstart, const int * const cellscount, 
			    const float rc,
			    const float XL, const float YL, const float ZL,
			    const float aij,
			    const float gamma,
			    const float sigma,
			    const float invsqrtdt,
			    const float seed1,
			    cudaStream_t stream);		  

void forces_dpd_cuda(const float * const xp, const float * const yp, const float * const zp,
		     const float * const xv, const float * const yv, const float * const zv,
		     float * const xa, float * const ya, float * const za,
		     const int np,
		     const float rc,
		     const float LX, const float LY, const float LZ,
		     const float a,
		     const float gamma,
		     const float sigma,
		     const float invsqrtdt,
		     const float seed);

/*
void forces_dpd_cuda_aos(float * const _xyzuvw, float * const _axayaz,
			 int * const order, const int np,
			 const float rc,
			 const float XL, const float YL, const float ZL,
			 const float aij,
			 const float gamma,
			 const float sigma,
			 const float invsqrtdt,
			 const float seed);
 */
void directforces_dpd_cuda_bipartite_nohost(
    const float * const xyzuvw, float * const axayaz, const int np,
    const float * const xyzuvw_src, const int np_src,
    const float aij, const float gamma, const float sigma, const float invsqrtdt,
    const float seed, const bool mask, cudaStream_t stream);

void forces_dpd_cuda_bipartite_nohost(cudaStream_t stream, const float2 * const xyzuvw, const int np, cudaTextureObject_t texDstStart,
				      cudaTextureObject_t texSrcStart, cudaTextureObject_t texSrcParticles, const int np_src,
				      const int3 halo_ncells,
				      const float aij, const float gamma, const float sigmaf,
				      const float seed, const bool mask, float * const axayaz);
