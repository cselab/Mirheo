
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

#include <core/datatypes.h>
#include <core/celllist.h>
#include <core/logger.h>
#include <core/components.h>
#include <core/xml/pugixml.hpp>

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


	std::string xml = R"(<node mass="1.0" density="2.0">)";
	pugi::xml_document config;
	config.load_string(xml.c_str());

	float3 length{64,32,55};
	float3 domainStart = -length / 2.0f;
	const float rc = 1.2f;
	ParticleVector dpds("dpd");
	CellList cells(&dpds, rc, length);

	InitialConditions ic = createIC(config.child("node"));
	ic.exec(MPI_COMM_WORLD, &dpds, {0,0,0}, length);

	const int np = dpds.np;
	HostBuffer<Particle> initial(np);
	auto initPtr = initial.hostPtr();
	for (int i=0; i<np; i++)
		initPtr[i] = dpds.coosvels[i];

	for (int i=0; i<50; i++)
		cells.build(0);

	dpds.coosvels.downloadFromDevice(true);

	HostBuffer<int> hcellsStart(cells.totcells+1);
	HostBuffer<uint8_t> hcellsSize(cells.totcells+1);

	hcellsStart.copy(cells.cellsStartSize, 0);
	hcellsSize. copy(cells.cellsSize, 0);

	HostBuffer<int> cellscount(cells.totcells+1);
	for (int i=0; i<cells.totcells+1; i++)
		cellscount[i] = 0;

	int total = 0;
	for (int pid=0; pid < initial.size(); pid++)
	{
		float3 coo{initial[pid].x[0], initial[pid].x[1], initial[pid].x[2]};
		float3 vel{initial[pid].u[0], initial[pid].u[1], initial[pid].u[2]};

		//vel += acc * dt;
		//coo += vel * dt;

		int actCid = cells.getCellId(coo);
		if (actCid >= 0)
		{
			cellscount[actCid]++;
			total++;
		}
	}

	printf("np = %d, vs reference  %d\n", dpds.np, total);
	for (int cid=0; cid < cells.totcells+1; cid++)
		if ( (hcellsStart[cid] >> cells.blendingPower) != cellscount[cid] )
			printf("cid %d:  %d (correct %d),  %d\n", cid, hcellsStart[cid] >> cells.blendingPower, cellscount[cid], hcellsStart[cid] & ((1<<cells.blendingPower) - 1));

	for (int cid=0; cid < cells.totcells; cid++)
	{
		const int start = hcellsStart[cid] & ((1<<cells.blendingPower) - 1);
		const int size = hcellsStart[cid] >> cells.blendingPower;
		for (int pid=start; pid < start + size; pid++)
		{
			const float3 cooDev{dpds.coosvels[pid].x[0], dpds.coosvels[pid].x[1], dpds.coosvels[pid].x[2]};
			const float3 velDev{dpds.coosvels[pid].u[0], dpds.coosvels[pid].u[1], dpds.coosvels[pid].u[2]};

			const int origId = dpds.coosvels[pid].i1;

			float3 coo{initial[origId].x[0], initial[origId].x[1], initial[origId].x[2]};
			float3 vel{initial[origId].u[0], initial[origId].u[1], initial[origId].u[2]};

//			vel += acc * dt;
//			coo += vel * dt;

			const float diff = std::max({
				fabs(coo.x - cooDev.x), fabs(coo.y - cooDev.y), fabs(coo.z - cooDev.z),
				fabs(vel.x - velDev.x), fabs(vel.y - velDev.y), fabs(vel.z - velDev.z) });

			int actCid = cells.getCellId<false>(cooDev);

			if (cid != actCid || diff > 1e-5)
				printf("cid  %d,  correct cid  %d  for pid %d:  [%e %e %e  %d]  correct: [%e %e %e  %d]\n",
						cid, actCid, pid, cooDev.x, cooDev.y, cooDev.z, dpds.coosvels[pid].i1,
						coo.x, coo.y, coo.z, initial[origId].i1);
		}
	}

	return 0;
}
