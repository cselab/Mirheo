// Yo ho ho ho
#define private public
#define protected public

#include <core/particle_vector.h>
#include <core/celllist.h>
#include <core/mpi/api.h>
#include <core/logger.h>
#include <core/integrate.h>
#include <core/interactions.h>
#include <core/components.h>
#include <core/xml/pugixml.hpp>

#include "timer.h"
#include <unistd.h>
#include <algorithm>

Logger logger;

void makeCells(const Particle* __restrict__ coos, Particle* __restrict__ buffer, int* __restrict__ cellsStartSize, int* __restrict__ cellsSize,
		int np, CellListInfo cinfo)
{
	for (int i=0; i<cinfo.totcells+1; i++)
		cellsSize[i] = 0;

	for (int i=0; i<np; i++)
		cellsSize[ cinfo.getCellId(make_float3(coos[i].r.x, coos[i].r.y, coos[i].r.z)) ]++;

	cellsStartSize[0] = 0;
	for (int i=1; i<=cinfo.totcells; i++)
		cellsStartSize[i] = cellsSize[i-1] + cellsStartSize[i-1];

	for (int i=0; i<np; i++)
	{
		const int cid = cinfo.getCellId(make_float3(coos[i].r.x, coos[i].r.y, coos[i].r.z));
		buffer[cellsStartSize[cid]] = coos[i];
		cellsStartSize[cid]++;
	}

	for (int i=0; i<cinfo.totcells; i++)
		cellsStartSize[i] -= cellsSize[i];
}

void integrate(Particle* __restrict__ coos, const Force* __restrict__ accs, int np, float dt, float3 domainStart, float3 length)
{
	for (int i=0; i<np; i++)
	{
		coos[i].u += accs[i].f*dt;
		coos[i].r += coos[i].u*dt;

		if (coos[i].r.x >  domainStart.x+length.x) coos[i].r.x -= length.x;
		if (coos[i].r.x <= domainStart.x)          coos[i].r.x += length.x;

		if (coos[i].r.y >  domainStart.y+length.y) coos[i].r.y -= length.y;
		if (coos[i].r.y <= domainStart.y)          coos[i].r.y += length.y;

		if (coos[i].r.z >  domainStart.z+length.z) coos[i].r.z -= length.z;
		if (coos[i].r.z <= domainStart.z)          coos[i].r.z += length.z;
	}
}


template<typename T>
T minabs(const T arg)
{
	return arg;
}

template<typename T, typename... Args>
T minabs(const T arg, const Args... other)
{
	const T v = minabs(other...	);
	return (std::abs(arg) < std::abs(v)) ? arg : v;
}

void forces(const Particle* __restrict__ coos, Force* __restrict__ accs, const int* __restrict__ cellsStartSize, const int* __restrict__ cellsSize,
		const CellListInfo& cinfo)
{

	const float ddt = 0.01;
	const float kBT = 1.0;
	const float gammadpd = 20;
	const float sigma = sqrt(2 * gammadpd * kBT);
	const float sigmaf = sigma / sqrt(ddt);
	const float aij = 50;

	const float3 length = cinfo.domainSize;

	auto addForce = [=] (int dstId, int srcId, Force& a)
	{
		float _xr = coos[dstId].r.x - coos[srcId].r.x;
		float _yr = coos[dstId].r.y - coos[srcId].r.y;
		float _zr = coos[dstId].r.z - coos[srcId].r.z;

		_xr = minabs(_xr, _xr - length.x, _xr + length.x);
		_yr = minabs(_yr, _yr - length.y, _yr + length.y);
		_zr = minabs(_zr, _zr - length.z, _zr + length.z);

		const float rij2 = _xr * _xr + _yr * _yr + _zr * _zr;

		if (rij2 > 1.0f) return;
		//assert(rij2 < 1);

		const float invrij = 1.0f / sqrt(rij2);
		const float rij = rij2 * invrij;
		const float argwr = 1.0f - rij;
		const float wr = argwr;

		const float xr = _xr * invrij;
		const float yr = _yr * invrij;
		const float zr = _zr * invrij;

		const float rdotv =
				xr * (coos[dstId].u.x - coos[srcId].u.x) +
				yr * (coos[dstId].u.y - coos[srcId].u.y) +
				zr * (coos[dstId].u.z - coos[srcId].u.z);

		const float myrandnr = 0;//Logistic::mean0var1(1, min(srcId, dstId), max(srcId, dstId));

		const float strength = aij * argwr - (gammadpd * wr * rdotv + sigmaf * myrandnr) * wr;

		a.f.x += strength * xr;
		a.f.y += strength * yr;
		a.f.z += strength * zr;

//		if (dstId == 8)
//			printf("%d -- %d  :  [%f %f %f %d] -- [%f %f %f %d] (dist2 %f) :  [%f %f %f]\n", dstId, srcId,
//					coos[dstId].r.x, coos[dstId].r.y, coos[dstId].r.z, coos[dstId].i1,
//					coos[srcId].r.x, coos[srcId].r.y, coos[srcId].r.z, coos[srcId].i1,
//					rij2, strength * xr, strength * yr, strength * zr);
	};


	const int3 ncells = cinfo.ncells;
	const int totcells = cinfo.totcells;

#pragma omp parallel for collapse(3)
	for (int cx = 0; cx < ncells.x; cx++)
		for (int cy = 0; cy < ncells.y; cy++)
			for (int cz = 0; cz < ncells.z; cz++)
			{
				const int cid = cinfo.encode(cx, cy, cz);

				for (int dstId = cellsStartSize[cid]; dstId < cellsStartSize[cid] + cellsSize[cid]; dstId++)
				{
					Force a {0,0,0,0};

					for (int dx = -1; dx <= 1; dx++)
						for (int dy = -1; dy <= 1; dy++)
							for (int dz = -1; dz <= 1; dz++)
							{
								int ncx, ncy, ncz;
								ncx = (cx+dx + ncells.x) % ncells.x;
								ncy = (cy+dy + ncells.y) % ncells.y;
								ncz = (cz+dz + ncells.z) % ncells.z;

								const int srcCid = cinfo.encode(ncx, ncy, ncz);
								if (srcCid >= totcells || srcCid < 0) continue;

								for (int srcId = cellsStartSize[srcCid]; srcId < cellsStartSize[srcCid] + cellsSize[srcCid]; srcId++)
								{
									if (dstId != srcId)
									{
										float _xr = coos[dstId].r.x - coos[srcId].r.x;
										float _yr = coos[dstId].r.y - coos[srcId].r.y;
										float _zr = coos[dstId].r.z - coos[srcId].r.z;

										_xr = minabs(_xr, _xr - length.x, _xr + length.x);
										_yr = minabs(_yr, _yr - length.y, _yr + length.y);
										_zr = minabs(_zr, _zr - length.z, _zr + length.z);

										const float rij2 = _xr * _xr + _yr * _yr + _zr * _zr;

										if (rij2 <= 1.0f)
										{
											const float invrij = 1.0f / sqrt(rij2);
											const float rij = rij2 * invrij;
											const float argwr = 1.0f - rij;
											const float wr = argwr;

											const float xr = _xr * invrij;
											const float yr = _yr * invrij;
											const float zr = _zr * invrij;

											const float rdotv =
													xr * (coos[dstId].u.x - coos[srcId].u.x) +
													yr * (coos[dstId].u.y - coos[srcId].u.y) +
													zr * (coos[dstId].u.z - coos[srcId].u.z);

											const float myrandnr = 0;//Logistic::mean0var1(1, min(srcId, dstId), max(srcId, dstId));

											const float strength = aij * argwr - (gammadpd * wr * rdotv + sigmaf * myrandnr) * wr;

											a.f.x += strength * xr;
											a.f.y += strength * yr;
											a.f.z += strength * zr;
										}
									}
								}
							}

					accs[dstId].f = a.f;
				}
			}
}

int main(int argc, char ** argv)
{
	// Init

	int provided;
//	MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
//	if (provided < MPI_THREAD_MULTIPLE)
//	{
//	    printf("ERROR: The MPI library does not have full thread support\n");
//	    MPI_Abort(MPI_COMM_WORLD, 1);
//	}

	MPI_Init(&argc, &argv);

	logger.init(MPI_COMM_WORLD, "manyranks.log", 9);

	if (argc < 4)
		die("Need 3 command line arguments");

	int nranks, rank;
	int ranks[] = {std::stoi(argv[1]), std::stoi(argv[2]), std::stoi(argv[3])};
	int periods[] = {1, 1, 1};
	int coords[3];
	MPI_Comm cartComm;


	MPI_Check( MPI_Comm_size(MPI_COMM_WORLD, &nranks) );
	MPI_Check( MPI_Comm_rank(MPI_COMM_WORLD, &rank) );
	MPI_Check( MPI_Cart_create(MPI_COMM_WORLD, 3, ranks, periods, 0, &cartComm) );
	MPI_Check( MPI_Cart_get(cartComm, 3, ranks, periods, coords) );


	std::string xml = R"(<node mass="1.0" density="8.0">)";
	pugi::xml_document config;
	config.load_string(xml.c_str());

	float3 length{32, 32, 32};
	float3 domainStart = -length / 2.0f;
	const float rc = 1.0f;
	ParticleVector dpds("dpd");
	CellList cells(&dpds, rc, length);

	InitialConditions ic = createIC(config.child("node"));
	ic.exec(MPI_COMM_WORLD, &dpds, {0,0,0}, length);

	const float k = 1;
	const float dt = 0.0025;
	const float kBT = 1.0;
	const float gammadpd = 20;
	const float sigmadpd = sqrt(2 * gammadpd * kBT);
	const float sigma_dt = sigmadpd / sqrt(dt);
	const float adpd = 50;

	auto inter = [=] (ParticleVector* pv, CellList* cl, const float t, cudaStream_t stream) {
		interactionDPD(InteractionType::Regular, pv, pv, cl, t, stream, adpd, gammadpd, sigma_dt, k, rc);
	};

	auto haloInt = [=] (ParticleVector* pv1, ParticleVector* pv2, CellList* cl, const float t, cudaStream_t stream) {
		interactionDPD(InteractionType::Halo, pv1, pv2, cl, t, stream, adpd, gammadpd, sigma_dt, k, rc);
	};

	dpds.local()->coosvels.downloadFromDevice(true);
	dpds.local()->forces.clear();
	int initialNP = dpds.local()->size();

	HostBuffer<Particle> locparticles(dpds.local()->size());
	for (int i=0; i<dpds.local()->size(); i++)
	{
		locparticles[i] = dpds.local()->coosvels[i];

		locparticles[i].r.x += (coords[0] + 0.5 - 0.5*ranks[0]) * length.x;
		locparticles[i].r.y += (coords[1] + 0.5 - 0.5*ranks[1]) * length.z;
		locparticles[i].r.z += (coords[2] + 0.5 - 0.5*ranks[2]) * length.y;
	}

	cudaStream_t defStream;
	CUDA_Check( cudaStreamCreateWithPriority(&defStream, cudaStreamNonBlocking, 10) );

	dpds.local()->pushStream(defStream);

	ParticleHaloExchanger halo(cartComm, defStream);
	halo.attach(&dpds, &cells);
	ParticleRedistributor redist(cartComm, defStream);
	redist.attach(&dpds, &cells);

	cells.makePrimary();
	cells.setStream(defStream);
	cells.build();

	const int niters = 5;

	printf("GPU execution\n");

	Timer tm;
	tm.start();

	for (int i=0; i<niters; i++)
	{
		cells.build();
		CUDA_Check( cudaStreamSynchronize(defStream) );

		dpds.local()->forces.clear();

		halo.init();
		inter(&dpds, &cells, dt*i, defStream);
		halo.finalize();

		haloInt(&dpds, &dpds, &cells, dt*i, defStream);

		integrateNoFlow(&dpds, dt, defStream);

		CUDA_Check( cudaStreamSynchronize(defStream) );

		redist.redistribute();
	}

	double elapsed = tm.elapsed() * 1e-9;

	printf("Finished in %f s, 1 step took %f ms\n", elapsed, elapsed / niters * 1000.0);

	if (argc < 5) return 0;
	cells.build();
	CUDA_Check( cudaStreamSynchronize(defStream) );


	dpds.local()->coosvels.downloadFromDevice(true);

	for (int i=0; i<dpds.local()->size(); i++)
	{
		dpds.local()->coosvels[i].r.x += (coords[0] + 0.5 - 0.5*ranks[0]) * length.x;
		dpds.local()->coosvels[i].r.y += (coords[1] + 0.5 - 0.5*ranks[1]) * length.z;
		dpds.local()->coosvels[i].r.z += (coords[2] + 0.5 - 0.5*ranks[2]) * length.y;
	}

	int totalParticles;
	int totalInitialPs;
	HostBuffer<int> sizes(ranks[0]*ranks[1]*ranks[2]), displs(ranks[0]*ranks[1]*ranks[2] + 1);
	HostBuffer<int> initialSizes(ranks[0]*ranks[1]*ranks[2]), initialDispls(ranks[0]*ranks[1]*ranks[2] + 1);

	MPI_Check( MPI_Gather(&dpds.local()->np, 1, MPI_INT, sizes.hostPtr(),        1, MPI_INT, 0, MPI_COMM_WORLD) );
	MPI_Check( MPI_Gather(&initialNP,        1, MPI_INT, initialSizes.hostPtr(), 1, MPI_INT, 0, MPI_COMM_WORLD) );

	displs[0] = 0;
	initialDispls[0] = 0;
	for (int i=1; i<ranks[0]*ranks[1]*ranks[2]+1; i++)
	{
		displs[i] = displs[i-1]+sizes[i-1];
		initialDispls[i] = initialDispls[i-1]+initialSizes[i-1];

		if (rank == 0) printf("Final:   %d %d %d \n", i-1, sizes[i-1], displs[i]);
		if (rank == 0) printf("Initial: %d %d %d \n", i-1, initialSizes[i-1], initialDispls[i]);
	}
	totalParticles = max(displs[displs.size() - 1], 0);
	totalInitialPs = max(initialDispls[initialDispls.size() - 1], 0);

	HostBuffer<Particle> finalParticles;
	HostBuffer<Particle> particles;

	if (rank == 0)
	{
		finalParticles.resize(totalParticles);
		particles.resize(totalInitialPs);
	}

	MPI_Datatype mpiPart;
	MPI_Check( MPI_Type_contiguous(sizeof(Particle), MPI_BYTE, &mpiPart) );
	MPI_Check( MPI_Type_commit(&mpiPart) );

	MPI_Check( MPI_Gatherv(dpds.local()->coosvels.hostPtr(), dpds.local()->size(), mpiPart, finalParticles.hostPtr(), sizes.hostPtr(),        displs.hostPtr(),        mpiPart, 0, MPI_COMM_WORLD) );
	MPI_Check( MPI_Gatherv(locparticles.hostPtr(),           initialNP,            mpiPart, particles.hostPtr(),      initialSizes.hostPtr(), initialDispls.hostPtr(), mpiPart, 0, MPI_COMM_WORLD) );


	if (rank == 0)
	{
		float3 globDomainSize = make_float3(length.x*ranks[0], length.y*ranks[1], length.z*ranks[2]);
		CellListInfo globalCells(1.0f, globDomainSize);

		HostBuffer<Particle> buffer(totalInitialPs);
		HostBuffer<Force> accs(totalInitialPs);
		HostBuffer<int>   cellsStartSize(globalCells.totcells+1), cellsSize(globalCells.totcells+1);


		printf("CPU execution\n");
		printf("NP:  %d,  ref  %d\n", totalParticles, totalInitialPs);

		for (int i=0; i<niters; i++)
		{
			printf("%d...", i);
			fflush(stdout);
			makeCells(particles.hostPtr(), buffer.hostPtr(), cellsStartSize.hostPtr(), cellsSize.hostPtr(), totalInitialPs, globalCells);
			containerSwap(particles, buffer);

			forces(particles.hostPtr(), accs.hostPtr(), cellsStartSize.hostPtr(), cellsSize.hostPtr(), globalCells);
			integrate(particles.hostPtr(), accs.hostPtr(), totalInitialPs, dt, -globDomainSize*0.5, globDomainSize);
		}

		printf("\nDone, checking\n");


		std::vector<int> gpuid(totalParticles), cpuid(totalInitialPs);
		for (int i=0; i<min(totalInitialPs, totalParticles); i++)
		{
			if (0 <= finalParticles[i].i1 && finalParticles[i].i1 < totalParticles)
				gpuid[finalParticles[i].i1] = i;
			else
			{
				printf("Wrong id on gpu: particle %d: [%f %f %f] %d\n", i,
					finalParticles[i].r.x, finalParticles[i].r.y, finalParticles[i].r.z, finalParticles[i].i1);
				return 0;
			}

			if (0 <= particles[i].i1 && particles[i].i1 < totalInitialPs)
				cpuid[particles[i].i1] = i;
			else
			{
				printf("Wrong id on cpu: particle %d: [%f %f %f] %d\n", i,
					particles[i].r.x, particles[i].r.y, particles[i].r.z, particles[i].i1);
				return 0;
			}
		}


		double l2 = 0, linf = -1;

		auto dot = [] (double3 a, double3 b) {
			return a.x*b.x + a.y*b.y + a.z*b.z;
		};

		auto make_double3 = [] (float3 v) {
			double3 res{v.x, v.y, v.z};
			return res;
		};

		for (int i=0; i<min(totalParticles, totalInitialPs); i++)
		{
			Particle cpuP = particles[cpuid[i]];
			Particle gpuP = finalParticles[gpuid[i]];

			double perr = -1;

			const double3 err = make_double3( fabs(cpuP.r - gpuP.r) + fabs(cpuP.u - gpuP.u) );
			linf = std::max({linf, err.x, err.y, err.z});
			perr = std::max({perr, err.x, err.y, err.z});
			l2 += dot(err, err);

			if (argc > 5 && perr > 0.01)
			{
				printf("id %8d diff %8e  [%12f %12f %12f  %8d] [%12f %12f %12f] \n"
						"                           ref [%12f %12f %12f  %8d] [%12f %12f %12f] [%12f %12f %12f] \n\n", i, perr,
						gpuP.r.x, gpuP.r.y, gpuP.r.z, gpuP.i1,
						gpuP.u.x, gpuP.u.y, gpuP.u.z,
						cpuP.r.x, cpuP.r.y, cpuP.r.z, cpuP.i1,
						cpuP.u.x, cpuP.u.y, cpuP.u.z, accs[cpuid[i]].f.x, accs[cpuid[i]].f.y, accs[cpuid[i]].f.z);
			}
		}

		l2 = sqrt(l2 / dpds.local()->size());
		printf("L2   norm: %f\n", l2);
		printf("Linf norm: %f\n", linf);
	}

	// Needed for passive waiting!
	if (rank == 0)
	{
		for (int i=1; i<nranks; i++)
			MPI_Check( MPI_Send(nullptr, 0, MPI_INT, i, 0, MPI_COMM_WORLD) );
	}
	else
	{
		MPI_Status stat;
		int recvd = 0;
		while (recvd == 0)
		{
			MPI_Check( MPI_Iprobe(0, 0, MPI_COMM_WORLD, &recvd, &stat) );
			if (recvd == 0) usleep(10000);
		}

		MPI_Check( MPI_Recv(nullptr, 0, MPI_INT, 0, 0, MPI_COMM_WORLD, &stat) );
	}


	MPI_Check( MPI_Finalize() );

	return 0;
}
