/*
 *  logistic.cu
 *  Part of CTC/mpi-dpd/
 *
 *  Created and authored by Diego Rossinelli on 2015-03-20.
 *  Copyright 2015. All rights reserved.
 *
 *  Users are NOT authorized
 *  to employ the present software for their own publications
 *  before getting a written permission from the author of this file.
 */

#include "logistic.h"
#include <cstdio>

// example usage of the logistic RNG

__global__ void foo( const float trunk, const uint nx, const uint ny )
{
	for(uint i=0;i<nx;i++) for(uint j=0;j<ny;j++) {
		printf("%f\n", mean0var1(trunk,i,j));
	}
}

int main() {
	KISS kiss( rand(), rand(), rand(), rand() );
	
	//for(int i=0;i<32;i++) printf("%f\n", kiss.get_float());

	foo<<<1,1>>>(kiss.get_float(),100,100);
	cudaDeviceSynchronize();

	return 0;
}
