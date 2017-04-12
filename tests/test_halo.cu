// Yo ho ho ho
#define private   public
#define protected public

#include <core/particle_vector.h>
#include <core/celllist.h>
#include <core/logger.h>
#include <core/xml/pugixml.hpp>
#include <core/components.h>

#include <core/mpi/api.h>

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
	logger.init(MPI_COMM_WORLD, "halo.log", 9);

	MPI_Check( MPI_Comm_size(MPI_COMM_WORLD, &nranks) );
	MPI_Check( MPI_Comm_rank(MPI_COMM_WORLD, &rank) );
	MPI_Check( MPI_Cart_create(MPI_COMM_WORLD, 3, ranks, periods, 0, &cartComm) );

	std::string xml = R"(<node mass="1.0" density="2.0">)";
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

	cells.build();

	dpds.coosvels.downloadFromDevice(true);

	cudaStream_t defStream = 0;

	ParticleHaloExchanger halo(cartComm, 0);
	halo.attach(&dpds, &cells);

	cells.build();
	CUDA_Check( cudaStreamSynchronize(defStream) );

	for (int i=0; i<10; i++)
	{
		halo.init();
		halo.finalize();
	}

	std::vector<Particle> bufs[27];
	dpds.coosvels.downloadFromDevice(true);
	dpds.halo.downloadFromDevice(true);

	for (int i=0; i<dpds.np; i++)
	{
		Particle& p = dpds.coosvels[i];

		int3 code = cells.getCellIdAlongAxis(p.r);
		int cx = code.x,  cy = code.y,  cz = code.z;
		auto ncells = cells.ncells;

		// 6
		if (cx == 0)          bufs[ (1*3 + 1)*3 + 0 ].push_back(addShift(p,  length.x,         0,         0));
		if (cx == ncells.x-1) bufs[ (1*3 + 1)*3 + 2 ].push_back(addShift(p, -length.x,         0,         0));
		if (cy == 0)          bufs[ (1*3 + 0)*3 + 1 ].push_back(addShift(p,         0,  length.y,         0));
		if (cy == ncells.y-1) bufs[ (1*3 + 2)*3 + 1 ].push_back(addShift(p,         0, -length.y,         0));
		if (cz == 0)          bufs[ (0*3 + 1)*3 + 1 ].push_back(addShift(p,         0,         0,  length.z));
		if (cz == ncells.z-1) bufs[ (2*3 + 1)*3 + 1 ].push_back(addShift(p,         0,         0, -length.z));

		// 12
		if (cx == 0          && cy == 0)          bufs[ (1*3 + 0)*3 + 0 ].push_back(addShift(p,  length.x,  length.y,         0));
		if (cx == ncells.x-1 && cy == 0)          bufs[ (1*3 + 0)*3 + 2 ].push_back(addShift(p, -length.x,  length.y,         0));
		if (cx == 0          && cy == ncells.y-1) bufs[ (1*3 + 2)*3 + 0 ].push_back(addShift(p,  length.x, -length.y,         0));
		if (cx == ncells.x-1 && cy == ncells.y-1) bufs[ (1*3 + 2)*3 + 2 ].push_back(addShift(p, -length.x, -length.y,         0));

		if (cy == 0          && cz == 0)          bufs[ (0*3 + 0)*3 + 1 ].push_back(addShift(p,         0,  length.y,  length.z));
		if (cy == ncells.y-1 && cz == 0)          bufs[ (0*3 + 2)*3 + 1 ].push_back(addShift(p,         0, -length.y,  length.z));
		if (cy == 0          && cz == ncells.z-1) bufs[ (2*3 + 0)*3 + 1 ].push_back(addShift(p,         0,  length.y, -length.z));
		if (cy == ncells.y-1 && cz == ncells.z-1) bufs[ (2*3 + 2)*3 + 1 ].push_back(addShift(p,         0, -length.y, -length.z));


		if (cz == 0          && cx == 0)          bufs[ (0*3 + 1)*3 + 0 ].push_back(addShift(p,  length.x,         0,  length.z));
		if (cz == ncells.z-1 && cx == 0)          bufs[ (2*3 + 1)*3 + 0 ].push_back(addShift(p,  length.x,         0, -length.z));
		if (cz == 0          && cx == ncells.x-1) bufs[ (0*3 + 1)*3 + 2 ].push_back(addShift(p, -length.x,         0,  length.z));
		if (cz == ncells.z-1 && cx == ncells.x-1) bufs[ (2*3 + 1)*3 + 2 ].push_back(addShift(p, -length.x,         0, -length.z));

		// 8
		if (cx == 0          && cy == 0          && cz == 0)          bufs[ (0*3 + 0)*3 + 0 ].push_back(addShift(p,  length.x,  length.y,  length.z));
		if (cx == 0          && cy == 0          && cz == ncells.z-1) bufs[ (2*3 + 0)*3 + 0 ].push_back(addShift(p,  length.x,  length.y, -length.z));
		if (cx == 0          && cy == ncells.y-1 && cz == 0)          bufs[ (0*3 + 2)*3 + 0 ].push_back(addShift(p,  length.x, -length.y,  length.z));
		if (cx == 0          && cy == ncells.y-1 && cz == ncells.z-1) bufs[ (2*3 + 2)*3 + 0 ].push_back(addShift(p,  length.x, -length.y, -length.z));
		if (cx == ncells.x-1 && cy == 0          && cz == 0)          bufs[ (0*3 + 0)*3 + 2 ].push_back(addShift(p, -length.x,  length.y,  length.z));
		if (cx == ncells.x-1 && cy == 0          && cz == ncells.z-1) bufs[ (2*3 + 0)*3 + 2 ].push_back(addShift(p, -length.x,  length.y, -length.z));
		if (cx == ncells.x-1 && cy == ncells.y-1 && cz == 0)          bufs[ (0*3 + 2)*3 + 2 ].push_back(addShift(p, -length.x, -length.y,  length.z));
		if (cx == ncells.x-1 && cy == ncells.y-1 && cz == ncells.z-1) bufs[ (2*3 + 2)*3 + 2 ].push_back(addShift(p, -length.x, -length.y, -length.z));
	}

	for (int i = 0; i<27; i++)
	{
		std::sort(bufs[i].begin(), bufs[i].end(), [] (Particle& a, Particle& b) { return a.i1 < b.i1; });

		std::sort((Particle*)halo.helpers[0]->sendBufs[i].hostPtr(), ((Particle*)halo.helpers[0]->sendBufs[i].hostPtr()) + halo.helpers[0]->counts[i],
				[] (Particle& a, Particle& b) { return a.i1 < b.i1; });

		if (bufs[i].size() != halo.helpers[0]->counts[i])
			printf("%2d-th halo differs in size: %5d, expected %5d\n", i, halo.helpers[0]->counts[i], (int)bufs[i].size());
		else
		{
			auto ptr = (Particle*)halo.helpers[0]->sendBufs[i].hostPtr();
			for (int pid = 0; pid < halo.helpers[0]->counts[i]; pid++)
			{
				const float diff = std::max({
					fabs(ptr[pid].r.x - bufs[i][pid].r.x),
					fabs(ptr[pid].r.y - bufs[i][pid].r.y),
					fabs(ptr[pid].r.z - bufs[i][pid].r.z) });

				if (bufs[i][pid].i1 != ptr[pid].i1 || diff > 1e-5)
					printf("Halo %2d:  %5d [%10.3e %10.3e %10.3e], expected %5d [%10.3e %10.3e %10.3e]\n",
							i, ptr[pid].i1, ptr[pid].r.x, ptr[pid].r.y, ptr[pid].r.z,
							bufs[i][pid].i1, bufs[i][pid].r.x, bufs[i][pid].r.y, bufs[i][pid].r.z);
			}
		}
	}

//	for (int i=0; i<dpds.halo.size(); i++)
//		printf("%d  %f %f %f\n", i, dpds.halo[i].r.x, dpds.halo[i].r.y, dpds.halo[i].r.z);

	return 0;
}
