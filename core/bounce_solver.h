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
template <typename Equation>
__device__ __host__ __inline__ float solveLinSearch(Equation F)
{
	// F is one dimensional equation
	// It returns value signed + or - depending on whether
	// coordinate is inside at the current time, or outside
	// Sign mapping to inside/outside is irrelevant

	const int   maxNIters = 20;
	const float tolerance = 1e-6f;

	float a = 0.0f;
	float b = 1.0f;
	float va = F(a);
	float vb = F(b);

	float mid;
	float vmid;

	// Check if the collision is there in the first place
	if (va*vb > 0.0f) return -1.0f;

	int iters;
	for (iters=0; iters<maxNIters; iters++)
	{
		const float lambda = min( max(vb / (vb - va),  0.1f), 0.9f );  // va*l + (1-l)*vb = 0
		mid = a *lambda + b *(1.0f - lambda);
		vmid = F(mid);

		if (va * vmid < 0.0f)
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
		return 0.0f;

	return mid;
}

