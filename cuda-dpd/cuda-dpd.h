    
void forces_dpd_cuda(float * const xyzuvw,
		     float * const axayaz, 
		     int * const order, const int np,
		     const float rc,
		     const float LX, const float LY, const float LZ,
		     const float a,
		     const float gamma,
		     const float sigma,
		     const float invsqrtdt,
		     float * const rsamples, int nsamples);
