#pragma once

void forces_sem_cuda(float * const xp, float * const yp, float * const zp,
		     float * const xv, float * const yv, float * const zv,
		     float * const xa, float * const ya, float * const za,
		     const int np,
		     const float rcutoff,
		     const float LX, const float LY, const float LZ,
		     const double gamma, const double temp, const double dt, const double u0, const double rho, const double req, const double D, const double rc);
   
