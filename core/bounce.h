/*
 * bounce.h
 *
 *  Created on: May 8, 2017
 *      Author: alexeedm
 */

#pragma once

template <typename InOutFunc>
__device__ __inline__ float bounceLinSearch(const float3 coo, const float3 vel, const float dt, InOutFunc F)
{
	const int   maxNIters = 20;
	const float tolerance = 5e-6;

	const float3 oldCoo = coo - dt*vel;

	float3 a = oldCoo;
	float3 b = coo;
	float3 mid;

	float va = F(a);
	float vb = F(b);
	float vmid;

	// Check if the collision is there in the first place
	if (va*vb >= 0.0f) return -1.0f;

	int iters;
	for (iters=0; iters<maxNIters; iters++)
	{
		const float lambda = min(max((vb / (vb - va)), 0.1f), 0.9f);  // va*l + (1-l)*vb = 0
		mid = a*lambda + b*(1.0f - lambda);
		vmid = F(mid);

		if (va * vmid < 0.0f)
		{
			vb = vmid;
			b = mid;
		}
		else
		{
			va = vmid;
			a = mid;
		}

		if (fabs(vmid) < tolerance) break;
	}

	// This shouldn't happen, but smth may go wrong
	// then will return 0 as this corresponds to the initial (outer) position
	if (fabs(vmid) > tolerance)
		return 0.0f;
	else
	{
		// Found intersection at alpha*dt
		// Take care if bounces are parallel to coo axes
		float alpha;
		if (oldCoo.x != coo.x)
			alpha = (oldCoo.x - mid.x) / (oldCoo.x - coo.x);
		else if (oldCoo.y != coo.y)
			alpha = (oldCoo.y - mid.y) / (oldCoo.y - coo.y);
		else if (oldCoo.z != coo.z)
			alpha = (oldCoo.z - mid.z) / (oldCoo.z - coo.z);
		else alpha = 1.0f;

		return alpha;
	}
}

