
/**
 * Copyright 1993-2012 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 */
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cassert>
#include <algorithm>

#include "../core/datatypes.h"
#include "../core/celllist.h"
#include "../core/logger.h"

Logger logger;

int main(int argc, char **argv)
{
	MPI_Init(&argc, &argv);
	logger.init(MPI_COMM_WORLD, "cells.log", 9);

	const int3 ncells{64, 64, 64};
	const float3 domainStart{-ncells.x / 2.0f, -ncells.y / 2.0f, -ncells.z / 2.0f};
	const int totcells = ncells.x*ncells.y*ncells.z;

	const int ndens = 12;
	int np = totcells*ndens;
	PinnedBuffer<Particle>   particles(np);
	PinnedBuffer<Particle>   out(np);

	PinnedBuffer<int> cellsStart(totcells + 1);
	PinnedBuffer<uint8_t> cellsSize(totcells + 1);

	srand48(0);

	printf("initializing...\n");

	int c = 0;
	for (int i=0; i<ncells.x; i++)
		for (int j=0; j<ncells.y; j++)
			for (int k=0; k<ncells.z; k++)
				for (int p=0; p<ndens * drand48(); p++)
				{
					particles[c].x[0] = i + drand48() + domainStart.x;
					particles[c].x[1] = j + drand48() + domainStart.y;
					particles[c].x[2] = k + drand48() + domainStart.z;
					particles[c].i1 = c;

					particles[c].u[0] = drand48() - 0.5;
					particles[c].u[1] = drand48() - 0.5;
					particles[c].u[2] = drand48() - 0.5;
					c++;
				}
	np = c;
	particles.resize(np);
	particles.synchronize(synchronizeDevice, 0);

	for (int i=0; i<50; i++)
		buildCellList((float4*)particles.devdata, np, domainStart, ncells, 1.0f, (float4*)out.devdata, cellsSize.devdata, cellsStart.devdata, 0);

	particles.synchronize(synchronizeHost, 0);
	out.synchronize(synchronizeHost, 0);
	cellsStart.synchronize(synchronizeHost, 0);
	cellsSize.synchronize(synchronizeHost, 0);

	HostBuffer<int> cellscount(totcells);
	for (int i=0; i<totcells; i++)
		cellscount[i] = 0;
	for (int pid=0; pid < np; pid++)
	{
		const float3 coo{particles[pid].x[0], particles[pid].x[1], particles[pid].x[2]};
		int actCid = getCellId(coo, domainStart, ncells, 1.0f);
		cellscount[actCid]++;
	}

	printf("np = %d\n", np);
//	for (int cid=0; cid < totcells+1; cid++)
//		if ( (int)cellsSize[cid] != cellscount[cid] ) printf("cid %d:  %d (%d),  %d\n", cid, (int)cellsSize[cid], cellscount[cid], cellsStart[cid]);

	for (int cid=0; cid < totcells; cid++)
		for (int pid=cellsStart[cid]; pid < cellsStart[cid+1]; pid++)
		{
			const float3 coo{out[pid].x[0], out[pid].x[1], out[pid].x[2]};
			const int origId = out[pid].i1;
			const float diff = std::max({ fabs(out[pid].x[0] - particles[origId].x[0]), fabs(out[pid].x[1] - particles[origId].x[1]), fabs(out[pid].x[2] - particles[origId].x[2]) });

			int actCid = getCellId(coo, domainStart, ncells, 1.0f);

			if (cid != actCid || diff > 1e-7)
				printf("cid  %d,  actCid  %d  for pid %d:  [%e %e %e], originally %d : [%e %e %e]\n",
						cid, actCid, pid, coo.x, coo.y, coo.z, origId, particles[origId].x[0], particles[origId].x[1], particles[origId].x[2]);
		}

	return 0;
}
