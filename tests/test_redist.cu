// Yo ho ho ho
#define private public

#include <core/containers.h>
#include <core/celllist.h>
#include <core/redistributor.h>
#include <core/logger.h>

Logger logger;

Particle addShift(Particle p, float a, float b, float c)
{
	Particle res = p;
	res.x[0] += a;
	res.x[1] += b;
	res.x[2] += c;

	return res;
}

int main(int argc, char ** argv)
{
	// Init

	int nranks, rank;
	int ranks[] = {1, 1, 1};
	int periods[] = {1, 1, 1};
	MPI_Comm cartComm;

	MPI_Init(&argc, &argv);
	logger.init(MPI_COMM_WORLD, "redist.log", 9);

	MPI_Check( MPI_Comm_size(MPI_COMM_WORLD, &nranks) );
	MPI_Check( MPI_Comm_rank(MPI_COMM_WORLD, &rank) );
	MPI_Check( MPI_Cart_create(MPI_COMM_WORLD, 3, ranks, periods, 0, &cartComm) );

	// Initial cells

	int3 ncells = {64, 64, 64};
	float3 domainStart = {-ncells.x / 2.0f, -ncells.y / 2.0f, -ncells.z / 2.0f};
	float3 length{(float)ncells.x, (float)ncells.y, (float)ncells.z};

	const int ndens = 8;

	ParticleVector dpds(ncells, domainStart, length);

	dpds.resize(dpds.totcells*ndens);

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

					dpds.coosvels[c].u[0] = 0;
					dpds.coosvels[c].u[1] = 0;
					dpds.coosvels[c].u[2] = 0;
					c++;
				}

	dpds.resize(c);
	dpds.coosvels.synchronize(synchronizeDevice);
	dpds.accs.clear();

	cudaStream_t defStream = 0;

	Redistributor redist(cartComm);
	redist.attach(&dpds, 7);

	buildCellList(dpds, defStream);
	CUDA_Check( cudaStreamSynchronize(defStream) );
	dpds.coosvels.synchronize(synchronizeHost);

	for (int i=0; i<dpds.np; i++)
	{
		dpds.coosvels[i].u[0] = (drand48() - 0.5);
		dpds.coosvels[i].u[1] = (drand48() - 0.5);
		dpds.coosvels[i].u[2] = (drand48() - 0.5);
	}

	dpds.coosvels.synchronize(synchronizeDevice, defStream);
	CUDA_Check( cudaStreamSynchronize(defStream) );

	const float dt = 0.5;

	for (int i=0; i<1; i++)
	{
		redist.redistribute(dt);
	}

	std::vector<Particle> bufs[27];
	for (int i=0; i<dpds.np; i++)
	{
		Particle& p = dpds.coosvels[i];
		float3 coo{p.x[0], p.x[1], p.x[2]};
		float3 vel{p.u[0], p.u[1], p.u[2]};

		coo += vel * dt;

		int cx = getCellIdAlongAxis<0, false>(coo.x, domainStart.x, ncells.x, 1.0f);
		int cy = getCellIdAlongAxis<1, false>(coo.y, domainStart.y, ncells.y, 1.0f);
		int cz = getCellIdAlongAxis<2, false>(coo.z, domainStart.z, ncells.z, 1.0f);

		//printf("%3d:  [%f %f %f]  %d %d %d\n", i, coo.x, coo.y, coo.z, cx, cy, cz);

		// 6
		if (cx == -1)         bufs[ (1*3 + 1)*3 + 0 ].push_back(addShift(p,  length.x,         0,         0));
		if (cx == ncells.x  ) bufs[ (1*3 + 1)*3 + 2 ].push_back(addShift(p, -length.x,         0,         0));
		if (cy == -1)         bufs[ (1*3 + 0)*3 + 1 ].push_back(addShift(p,         0,  length.y,         0));
		if (cy == ncells.y  ) bufs[ (1*3 + 2)*3 + 1 ].push_back(addShift(p,         0, -length.y,         0));
		if (cz == -1)         bufs[ (0*3 + 1)*3 + 1 ].push_back(addShift(p,         0,         0,  length.z));
		if (cz == ncells.z  ) bufs[ (2*3 + 1)*3 + 1 ].push_back(addShift(p,         0,         0, -length.z));

		// 12
		if (cx == -1         && cy == -1)         bufs[ (1*3 + 0)*3 + 0 ].push_back(addShift(p,  length.x,  length.y,         0));
		if (cx == ncells.x   && cy == -1)         bufs[ (1*3 + 0)*3 + 2 ].push_back(addShift(p, -length.x,  length.y,         0));
		if (cx == -1         && cy == ncells.y  ) bufs[ (1*3 + 2)*3 + 0 ].push_back(addShift(p,  length.x, -length.y,         0));
		if (cx == ncells.x   && cy == ncells.y  ) bufs[ (1*3 + 2)*3 + 2 ].push_back(addShift(p, -length.x, -length.y,         0));

		if (cy == -1         && cz == -1)         bufs[ (0*3 + 0)*3 + 1 ].push_back(addShift(p,         0,  length.y,  length.z));
		if (cy == ncells.y   && cz == -1)         bufs[ (0*3 + 2)*3 + 1 ].push_back(addShift(p,         0, -length.y,  length.z));
		if (cy == -1         && cz == ncells.z  ) bufs[ (2*3 + 0)*3 + 1 ].push_back(addShift(p,         0,  length.y, -length.z));
		if (cy == ncells.y   && cz == ncells.z  ) bufs[ (2*3 + 2)*3 + 1 ].push_back(addShift(p,         0, -length.y, -length.z));


		if (cz == -1         && cx == -1)         bufs[ (0*3 + 1)*3 + 0 ].push_back(addShift(p,  length.x,         0,  length.z));
		if (cz == ncells.z   && cx == -1)         bufs[ (2*3 + 1)*3 + 0 ].push_back(addShift(p,  length.x,         0, -length.z));
		if (cz == -1         && cx == ncells.x  ) bufs[ (0*3 + 1)*3 + 2 ].push_back(addShift(p, -length.x,         0,  length.z));
		if (cz == ncells.z   && cx == ncells.x  ) bufs[ (2*3 + 1)*3 + 2 ].push_back(addShift(p, -length.x,         0, -length.z));

		// 8
		if (cx == -1         && cy == -1         && cz == -1)         bufs[ (0*3 + 0)*3 + 0 ].push_back(addShift(p,  length.x,  length.y,  length.z));
		if (cx == -1         && cy == -1         && cz == ncells.z  ) bufs[ (2*3 + 0)*3 + 0 ].push_back(addShift(p,  length.x,  length.y, -length.z));
		if (cx == -1         && cy == ncells.y   && cz == -1)         bufs[ (0*3 + 2)*3 + 0 ].push_back(addShift(p,  length.x, -length.y,  length.z));
		if (cx == -1         && cy == ncells.y   && cz == ncells.z  ) bufs[ (2*3 + 2)*3 + 0 ].push_back(addShift(p,  length.x, -length.y, -length.z));
		if (cx == ncells.x   && cy == -1         && cz == -1)         bufs[ (0*3 + 0)*3 + 2 ].push_back(addShift(p, -length.x,  length.y,  length.z));
		if (cx == ncells.x   && cy == -1         && cz == ncells.z  ) bufs[ (2*3 + 0)*3 + 2 ].push_back(addShift(p, -length.x,  length.y, -length.z));
		if (cx == ncells.x   && cy == ncells.y   && cz == -1)         bufs[ (0*3 + 2)*3 + 2 ].push_back(addShift(p, -length.x, -length.y,  length.z));
		if (cx == ncells.x   && cy == ncells.y   && cz == ncells.z  ) bufs[ (2*3 + 2)*3 + 2 ].push_back(addShift(p, -length.x, -length.y, -length.z));
	}

	for (int i = 0; i<27; i++)
	{
		std::sort(bufs[i].begin(), bufs[i].end(), [] (Particle& a, Particle& b) { return a.i1 < b.i1; });

		std::sort(redist.helpers[0].sendBufs[i].hostdata, redist.helpers[0].sendBufs[i].hostdata + redist.helpers[0].counts[i],
				[] (PandA& a, PandA& b) { return a.p.i1 < b.p.i1; });

		if (bufs[i].size() != redist.helpers[0].counts[i])
			printf("%2d-th redist differs in size: %5d, expected %5d\n", i, redist.helpers[0].counts[i], (int)bufs[i].size());

		for (int pid = 0; pid < std::min(redist.helpers[0].counts[i], (int)bufs[i].size()); pid++)
		{
			const float diff = std::max({
				fabs(redist.helpers[0].sendBufs[i][pid].p.x[0] - bufs[i][pid].x[0]),
				fabs(redist.helpers[0].sendBufs[i][pid].p.x[1] - bufs[i][pid].x[1]),
				fabs(redist.helpers[0].sendBufs[i][pid].p.x[2] - bufs[i][pid].x[2]) });

			if (bufs[i][pid].i1 != redist.helpers[0].sendBufs[i][pid].p.i1 || diff > 1e-5)
				printf("redist %2d:  %5d [%10.3e %10.3e %10.3e], expected %5d [%10.3e %10.3e %10.3e]\n",
						i, redist.helpers[0].sendBufs[i][pid].p.i1, redist.helpers[0].sendBufs[i][pid].p.x[0],
						redist.helpers[0].sendBufs[i][pid].p.x[1], redist.helpers[0].sendBufs[i][pid].p.x[2],
						bufs[i][pid].i1, bufs[i][pid].x[0], bufs[i][pid].x[1], bufs[i][pid].x[2]);
		}
	}

	//for (int i=0; i<dpds.redist.size; i++)
	//	printf("%d  %f %f %f\n", i, dpds.redist[i].x[0], dpds.redist[i].x[1], dpds.redist[i].x[2]);

	return 0;
}
