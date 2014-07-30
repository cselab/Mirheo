#pragma once

void forces_dpd_cuda(float * const xp, float * const yp, float * const zp,
		     float * const xv, float * const yv, float * const zv,
		     float * const xa, float * const ya, float * const za,
		     const int np,
		     const float rc,
		     const float LX, const float LY, const float LZ,
		     const float a,
		     const float gamma,
		     const float sigma,
		     const float invsqrtdt);
    
void forces_dpd_cuda_bipartite(float * const xp1, float * const yp1, float * const zp1,
			       float * const xv1, float * const yv1, float * const zv1,
			       float * const xa1, float * const ya1, float * const za1,
			       const int np1, const int gp1id_start,
			       float * const xp2, float * const yp2, float * const zp2,
			       float * const xv2, float * const yv2, float * const zv2,
			       float * const xa2, float * const ya2, float * const za2,
			       const int np2, const int gp2id_start,
			       const float rc, const float LX, const float LY, const float LZ,
			       const float a, const float gamma, const float sigma, const float invsqrtdt);

