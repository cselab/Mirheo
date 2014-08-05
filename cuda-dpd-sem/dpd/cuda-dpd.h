#pragma once

void forces_dpd_cuda(const float * const xp, const float * const yp, const float * const zp,
		     const float * const xv, const float * const yv, const float * const zv,
		     float * const xa, float * const ya, float * const za,
		     const int np,
		     const float rc,
		     const float LX, const float LY, const float LZ,
		     const float a,
		     const float gamma,
		     const float sigma,
		     const float invsqrtdt);
    
void forces_dpd_cuda_bipartite(const float * const xp1, const float * const yp1, const float * const zp1,
			       const float * const xv1, const float * const yv1, const float * const zv1,
			       float * const xa1, float * const ya1,  float * const za1,
			       const int np1, const int gp1id_start,	 
			       const float * const xp2, const float * const yp2, const float * const zp2,
			       const float * const xv2, const float * const yv2, const float * const zv2,
			       float * const xa2,  float * const ya2,  float * const za2,
			       const int np2, const int gp2id_start,
			       const float rc, const float LX, const float LY, const float LZ,
			       const float a, const float gamma, const float sigma, const float invsqrtdt);

