
/**
 * Copyright 1993-2012 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 */

// Yo ho ho ho
#define private   public
#define protected public

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cassert>
#include <algorithm>

#include <core/pvs/particle_vector.h>
#include <core/celllist.h>
#include <core/logger.h>
#include <core/initial_conditions/uniform.h>

Logger logger;

int main(int argc, char **argv)
{
	int nranks, rank;
	int ranks[] = {1, 1, 1};
	int periods[] = {1, 1, 1};
	MPI_Comm cartComm;

	MPI_Init(&argc, &argv);

	logger.init(MPI_COMM_WORLD, "cells.log", 9);

	MPI_Check( MPI_Comm_size(MPI_COMM_WORLD, &nranks) );
	MPI_Check( MPI_Comm_rank(MPI_COMM_WORLD, &rank) );
	MPI_Check( MPI_Cart_create(MPI_COMM_WORLD, 3, ranks, periods, 0, &cartComm) );


	float3 length{64,64,64};
	float3 domainStart = -length / 2.0f;
	const float rc = 1.2f;
	ParticleVector dpds("dpd", 1.0f);
	CellList *cells = new PrimaryCellList(&dpds, rc, length);

	UniformIC ic(8);
	ic.exec(MPI_COMM_WORLD, &dpds, {0,0,0}, length, 0);

	const int np = dpds.local()->size();
	HostBuffer<Particle> initial(np);
	auto initPtr = initial.hostPtr();
	for (int i=0; i<np; i++)
		initPtr[i] = dpds.local()->coosvels[i];

	for (int i=0; i<50; i++)
	{
		cells->build(0);
		dpds.local()->changedStamp++;
	}

	dpds.local()->coosvels.downloadFromDevice(0, true);

	HostBuffer<int> hcellsStart(cells->totcells+1);
	HostBuffer<int> hcellsSize(cells->totcells+1);

	hcellsStart.copy(cells->cellStarts, 0);
	hcellsSize. copy(cells->cellSizes, 0);

	HostBuffer<int> cellscount(cells->totcells+1);
	for (int i=0; i<cells->totcells+1; i++)
		cellscount[i] = 0;

	int total = 0;
	for (int pid=0; pid < initial.size(); pid++)
	{
		float3 coo{initial[pid].r.x, initial[pid].r.y, initial[pid].r.z};
		float3 vel{initial[pid].u.x, initial[pid].u.y, initial[pid].u.z};

		//vel += acc * dt;
		//coo += vel * dt;

		int actCid = cells->getCellId(coo);
		if (actCid >= 0)
		{
			cellscount[actCid]++;
			total++;
		}
	}

	printf("np = %d, vs reference  %d\n", dpds.local()->size(), total);
	for (int cid=0; cid < cells->totcells+1; cid++)
		if ( (hcellsSize[cid]) != cellscount[cid] )
			printf("cid %d:  %d (correct %d),  %d\n",
					cid, hcellsSize[cid], cellscount[cid], hcellsStart[cid]);

	for (int cid=0; cid < cells->totcells; cid++)
	{
		const int start = hcellsStart[cid];
		const int size = hcellsSize[cid];
		for (int pid=start; pid < start + size; pid++)
		{
			const float3 cooDev{dpds.local()->coosvels[pid].r.x, dpds.local()->coosvels[pid].r.y, dpds.local()->coosvels[pid].r.z};
			const float3 velDev{dpds.local()->coosvels[pid].u.x, dpds.local()->coosvels[pid].u.y, dpds.local()->coosvels[pid].u.z};

			const int origId = dpds.local()->coosvels[pid].i1;

			float3 coo{initial[origId].r.x, initial[origId].r.y, initial[origId].r.z};
			float3 vel{initial[origId].u.x, initial[origId].u.y, initial[origId].u.z};

//			vel += acc * dt;
//			coo += vel * dt;

			const float diff = std::max({
				fabs(coo.x - cooDev.x), fabs(coo.y - cooDev.y), fabs(coo.z - cooDev.z),
				fabs(vel.x - velDev.x), fabs(vel.y - velDev.y), fabs(vel.z - velDev.z) });

			int actCid = cells->getCellId<false>(cooDev);

			if (cid != actCid || diff > 1e-5)
				printf("cid  %d,  correct cid  %d  for pid %d:  [%e %e %e  %d]  correct: [%e %e %e  %d]\n",
						cid, actCid, pid, cooDev.x, cooDev.y, cooDev.z, dpds.local()->coosvels[pid].i1,
						coo.x, coo.y, coo.z, initial[origId].i1);
		}
	}

	return 0;
}
