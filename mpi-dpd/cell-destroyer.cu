/*
 *  cell-destroyer.cpp
 *  ctc-garbald
 *
 *  Created by Dmitry Alexeev on Mar 22, 2015
 *  Copyright 2015 ETH Zurich. All rights reserved.
 *
 */


#include "cell-destroyer.h"

inline float CellDestroyer::toGlobal(float x)
{
	return (xcoo+0.5)*XSIZE_SUBDOMAIN + x;
}

CellDestroyer::CellDestroyer(float xlim, int xcoo) : xlim(xlim), xcoo(xcoo) {}

int CellDestroyer::destroy(ParticleArray& p, CollectionRBC* rbcs)
{
	if (rbcs == NULL) return 0;

	PinnedHostBuffer<CudaRBC::Extent> extents(rbcs->count() + 1);

	CudaRBC::extent_nohost(0, rbcs->count(), (float*)rbcs->data(), extents.devptr);
	CudaRBC::extent_nohost(0, 1, CudaRBC::get_orig_xyzuvw(), extents.devptr + rbcs->count());
	cudaDeviceSynchronize();

	int left = 0, right = rbcs->count() - 1;
	int oldsize = rbcs->count();

	while (left < right)
	{
//		printf("l %d  r %d\n", left, right);
//		printf("%f   %f\n", toGlobal(extents.data[left].xmax), xlim);

		while (left <= right && toGlobal(extents.data[left].xmax)  < xlim)
			left++;

		while (left <= right && toGlobal(extents.data[right].xmax) > xlim)
			right--;

		if (left < right)
		{
			printf("Killing!! Moved: %d  =>  %d\n", right, left);
			CUDA_CHECK( cudaMemcpyAsync(rbcs->data() + left*rbcs->nvertices,
					                    rbcs->data() + right*rbcs->nvertices,
										6*rbcs->nvertices*sizeof(float), cudaMemcpyDeviceToDevice) );
			left++;
			right--;
		}
	}
	rbcs->preserve_resize(right+1);

	return oldsize - rbcs->count();
}
