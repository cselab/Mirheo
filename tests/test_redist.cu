// Yo ho ho ho
#define private public
#define protected public

#include <core/particle_vector.h>
#include <core/celllist.h>
#include <core/mpi/api.h>
#include <core/logger.h>

#include <core/xml/pugixml.hpp>
#include <core/components.h>

Logger logger;

Particle addShift(Particle p, float a, float b, float c)
{
	Particle res = p;
	res.r.x += a;
	res.r.y += b;
	res.r.z += c;

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

	std::string xml = R"(<node mass="1.0" density="8.0">)";
	pugi::xml_document config;
	config.load_string(xml.c_str());

	float3 length{24,24,24};
	float3 domainStart = -length / 2.0f;
	const float rc = 1.0f;
	ParticleVector dpds("dpd");
	CellList cells(&dpds, rc, length);
	cells.setStream(0);
	cells.makePrimary();

	InitialConditions ic = createIC(config.child("node"));
	ic.exec(MPI_COMM_WORLD, &dpds, {0,0,0}, length);

	const int initialNP = dpds.np;
	HostBuffer<Particle> host(dpds.np);
	const float dt = 0.1;
	for (int i=0; i<dpds.np; i++)
	{
		dpds.coosvels[i].u.z = 5*(drand48() - 0.5);
		dpds.coosvels[i].u.y = 5*(drand48() - 0.5);
		dpds.coosvels[i].u.z = 5*(drand48() - 0.5);

		dpds.coosvels[i].r += dt * dpds.coosvels[i].u;

		host[i] = dpds.coosvels[i];
	}

	dpds.coosvels.uploadToDevice();

	cudaStream_t defStream = 0;

	ParticleRedistributor redist(cartComm, defStream);
	cells.build();
	redist.attach(&dpds, &cells);
	CUDA_Check( cudaStreamSynchronize(defStream) );

	for (int i=0; i<1; i++)
	{
		redist.redistribute();
	}

	std::vector<Particle> bufs[27];

	for (int i=0; i<initialNP; i++)
	{
		Particle& p = host[i];

		int3 code = cells.getCellIdAlongAxis<false>(p.r);
		int cx = code.x,  cy = code.y,  cz = code.z;
		auto ncells = cells.ncells;

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

		std::sort((Particle*)redist.helpers[0]->sendBufs[i].hostPtr(), ((Particle*)redist.helpers[0]->sendBufs[i].hostPtr()) + redist.helpers[0]->counts[i],
				[] (Particle& a, Particle& b) { return a.i1 < b.i1; });

		if (bufs[i].size() != redist.helpers[0]->counts[i])
			printf("%2d-th redist differs in size: %5d, expected %5d\n", i, redist.helpers[0]->counts[i], (int)bufs[i].size());
		else
		{
			auto ptr = (Particle*)redist.helpers[0]->sendBufs[i].hostPtr();
			for (int pid = 0; pid < redist.helpers[0]->counts[i]; pid++)
			{
				const float diff = std::max({
					fabs(ptr[pid].r.x - bufs[i][pid].r.x),
							fabs(ptr[pid].r.y - bufs[i][pid].r.y),
							fabs(ptr[pid].r.z - bufs[i][pid].r.z) });

				if (bufs[i][pid].i1 != ptr[pid].i1 || diff > 1e-5)
					printf("redist %2d:  %5d [%10.3e %10.3e %10.3e], expected %5d [%10.3e %10.3e %10.3e]\n",
							i, ptr[pid].i1, ptr[pid].r.x, ptr[pid].r.y, ptr[pid].r.z,
							bufs[i][pid].i1, bufs[i][pid].r.x, bufs[i][pid].r.y, bufs[i][pid].r.z);
			}
		}
	}

	return 0;
}
