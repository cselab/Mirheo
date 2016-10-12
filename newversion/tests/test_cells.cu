
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
	int nranks, rank;
	int ranks[] = {1, 1, 1};
	int periods[] = {1, 1, 1};
	MPI_Comm cartComm;

	int provided;
	MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
	if (provided < MPI_THREAD_MULTIPLE)
	{
		printf("ERROR: The MPI library does not have full thread support\n");
		MPI_Abort(MPI_COMM_WORLD, 1);
	}

	logger.init(MPI_COMM_WORLD, "cells.log", 9);

	MPI_Check( MPI_Comm_size(MPI_COMM_WORLD, &nranks) );
	MPI_Check( MPI_Comm_rank(MPI_COMM_WORLD, &rank) );
	MPI_Check( MPI_Cart_create(MPI_COMM_WORLD, 3, ranks, periods, 0, &cartComm) );

	// Initial cells

	int3 ncells = {64, 64, 64};
	float3 domainStart = {-ncells.x / 2.0f, -ncells.y / 2.0f, -ncells.z / 2.0f};
	float3 length{(float)ncells.x, (float)ncells.y, (float)ncells.z};
	ParticleVector dpds(ncells, domainStart, length);
	const float dt = 0.1;

	const int ndens = 8;
	dpds.resize(ncells.x*ncells.y*ncells.z * ndens);

	srand48(0);

	printf("initializing...\n");

	int c = 0;
	for (int i=0; i<ncells.x; i++)
		for (int j=0; j<ncells.y; j++)
			for (int k=0; k<ncells.z; k++)
				for (int p=0; p<ndens; p++)
				{
					dpds.coosvels[c].x[0] = i + drand48() + domainStart.x;
					dpds.coosvels[c].x[1] = j + drand48() + domainStart.y;
					dpds.coosvels[c].x[2] = k + drand48() + domainStart.z;
					dpds.coosvels[c].i1 = c;

					dpds.coosvels[c].u[0] = (drand48() - 0.5);
					dpds.coosvels[c].u[1] = (drand48() - 0.5);
					dpds.coosvels[c].u[2] = (drand48() - 0.5);
					c++;
				}

	int np = c;

	dpds.resize(np, resizePreserve);
	dpds.accs.clear();
	dpds.coosvels.synchronize(synchronizeDevice);

	HostBuffer<Particle> initial(np);
	HostBuffer<Acceleration> accInit(np);

	for (int i=0; i<np; i++)
	{
		accInit[i].a[0] = drand48() - 0.5;
		accInit[i].a[1] = drand48() - 0.5;
		accInit[i].a[2] = drand48() - 0.5;

		initial[i] = dpds.coosvels[i];
	}

	dpds.accs.copy(accInit);

	for (int i=0; i<1; i++)
		buildCellListAndIntegrate(dpds, dt, 0);

	dpds.coosvels.synchronize(synchronizeHost);
	cudaDeviceSynchronize();


	HostBuffer<int> hcellsStart(dpds.totcells+1);
	HostBuffer<uint8_t> hcellsSize(dpds.totcells+1);

	hcellsStart.copy(dpds.cellsStart);
	hcellsSize. copy(dpds.cellsSize);

	HostBuffer<int> cellscount(dpds.totcells+1);
	for (int i=0; i<dpds.totcells; i++)
		cellscount[i] = 0;

	int total = 0;
	for (int pid=0; pid < initial.size; pid++)
	{
		float3 coo{initial[pid].x[0], initial[pid].x[1], initial[pid].x[2]};
		float3 vel{initial[pid].u[0], initial[pid].u[1], initial[pid].u[2]};
		float3 acc{accInit[pid].a[0], accInit[pid].a[1], accInit[pid].a[2]};

		vel += acc * dt;
		coo += vel * dt;

		int actCid = getCellId<false>(coo, domainStart, ncells, 1.0f);
		if (actCid >= 0)
		{
			cellscount[actCid]++;
			total++;
		}
	}

	printf("np = %d, vs reference  %d\n", dpds.np, total);
	for (int cid=0; cid < dpds.totcells+1; cid++)
		if ( (hcellsStart[cid] >> 26) != cellscount[cid] )
			printf("cid %d:  %d (%d),  %d\n", cid, hcellsStart[cid] >> 26, cellscount[cid], hcellsStart[cid] & ((1<<26) - 1));

	for (int cid=0; cid < dpds.totcells; cid++)
	{
		const int start = hcellsStart[cid] & ((1<<26) - 1);
		const int size = hcellsStart[cid] >> 26;
		for (int pid=start; pid < start + size; pid++)
		{
			const float3 cooDev{dpds.coosvels[pid].x[0], dpds.coosvels[pid].x[1], dpds.coosvels[pid].x[2]};
			const float3 velDev{dpds.coosvels[pid].u[0], dpds.coosvels[pid].u[1], dpds.coosvels[pid].u[2]};

			const int origId = dpds.coosvels[pid].i1;

			float3 coo{initial[origId].x[0], initial[origId].x[1], initial[origId].x[2]};
			float3 vel{initial[origId].u[0], initial[origId].u[1], initial[origId].u[2]};
			float3 acc{accInit[origId].a[0], accInit[origId].a[1], accInit[origId].a[2]};

			vel += acc * dt;
			coo += vel * dt;

			const float diff = std::max({
				fabs(coo.x - cooDev.x), fabs(coo.y - cooDev.y), fabs(coo.z - cooDev.z),
				fabs(vel.x - velDev.x), fabs(vel.y - velDev.y), fabs(vel.z - velDev.z) });

			int actCid = getCellId<false>(cooDev, domainStart, ncells, 1.0f);

			if (cid != actCid || diff > 1e-5)
				printf("cid  %d,  correct cid  %d  for pid %d:  [%e %e %e  %d] [%e %e %e] correct: [%e %e %e  %d] [%e %e %e]\n",
						cid, actCid, pid, cooDev.x, cooDev.y, cooDev.z, dpds.coosvels[pid].i1, velDev.x, velDev.y, velDev.z,
						coo.x, coo.y, coo.z, initial[origId].i1, vel.x, vel.y, vel.z);
		}
	}

	return 0;
}
