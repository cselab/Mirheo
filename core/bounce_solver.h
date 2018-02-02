/*
 * bounce.h
 *
 *  Created on: May 8, 2017
 *      Author: alexeedm
 */

#pragma once

/**
 * Find alpha such that F( alpha ) = 0, 0 <= alpha <= 1
 */
template <typename Real, typename Equation>
__device__ inline Real _solveLinSearch_templ(Equation F, Real tolerance)
{
	// F is one dimensional equation
	// It returns value signed + or - depending on whether
	// coordinate is inside at the current time, or outside
	// Sign mapping to inside/outside is irrelevant

	const int maxNIters = 20;

	Real a = (Real)0.0;
	Real b = (Real)1.0;
	Real va = F(a);
	Real vb = F(b);

	Real mid, vmid;

	// Check if the collision is there in the first place
	if (va*vb > (Real)0.0) return (Real)-1.0;

	int iters;
	for (iters=0; iters<maxNIters; iters++)
	{
		const Real lambda = min( max(vb / (vb - va),  (Real)0.1), (Real)0.9 );  // va*l + (1-l)*vb = 0
		mid = a *lambda + b *((Real)1.0 - lambda);
		vmid = F(mid);

		if (va * vmid < (Real)0.0)
		{
			vb = vmid;
			b  = mid;
		}
		else
		{
			va = vmid;
			a = mid;
		}

		if (fabs(vmid) < tolerance)
			break;
	}

	// This shouldn't happen, but smth may go wrong
	// then will return 0 as this corresponds to the initial (outer) position
	if (fabs(vmid) > tolerance)
		return (Real)0.0;

	return mid;
}

template <typename Equation>
__device__ inline float solveLinSearch(Equation F, float tolerance = 1e-6f)
{
	return _solveLinSearch_templ<float>(F, tolerance);
}

template <typename Equation>
__device__ inline double solveLinSearch_double(Equation F, double tolerance = 1e-10)
{
	return _solveLinSearch_templ<double>(F, tolerance);
}

