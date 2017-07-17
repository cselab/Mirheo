/*
 * bounce.h
 *
 *  Created on: May 8, 2017
 *      Author: alexeedm
 */

#pragma once

template <typename InOutFunc>
__device__ __inline__ float bounceLinSearch(const float3 oldCoo, const float3 coo, InOutFunc F)
{
	// F is the function given 3D coordinate and time (0 to 1)
	// returns value signed + or - depending on whether
	// coordinate is inside at the current time, or outside
	// Sign mapping to inside/outside is irrelevant

	const int   maxNIters = 20;
	const float tolerance = 5e-6;

	float3 a = oldCoo;
	float3 b = coo;
	float3 mid;

	float la = 0.0f;
	float lb = 1;
	float l = 0;

	float va = F(a);
	float vb = F(b);
	float vmid;

	// Check if the collision is there in the first place
	if (va*vb >= 0.0f) return -1.0f;

	int iters;
	for (iters=0; iters<maxNIters; iters++)
	{
		const float lambda = min(max((vb / (vb - va)), 0.1f), 0.9f);  // va*l + (1-l)*vb = 0
		mid = a *lambda + b *(1.0f - lambda);
		l   = la*lambda + lb*(1.0f - lambda);
		vmid = F(mid);

		if (va * vmid < 0.0f)
		{
			lb = l;
			vb = vmid;
			b  = mid;
		}
		else
		{
			la = l;
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

	// Make sure that the sign is the same as on the left (oldCoo) side
	if (va*vmid > 0.0f)
		return l;
	else
	{
		do {
			l -= tolerance;
		} while (F(oldCoo + l*(coo-oldCoo)) * va < 0.0f);

		return l;
	}
}

