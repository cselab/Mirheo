// Yo ho ho ho
#define private public

#include <core/containers.h>
#include <core/celllist.h>
#include <core/halo_exchanger.h>
#include <core/redistributor.h>
#include <core/logger.h>
#include <core/integrate.h>
#include <core/interactions.h>
#include <core/components.h>
#include <core/xml/pugixml.hpp>

#include "timer.h"
#include <unistd.h>

Logger logger;

void makeCells(const Particle* __restrict__ coos, Particle* __restrict__ buffer, int* __restrict__ cellsStart, int* __restrict__ cellsSize,
		int np, CellListInfo cinfo)
{
	for (int i=0; i<cinfo.totcells+1; i++)
		cellsSize[i] = 0;

	for (int i=0; i<np; i++)
		cellsSize[ cinfo.getCellId(make_float3(coos[i].x[0], coos[i].x[1], coos[i].x[2])) ]++;

	cellsStart[0] = 0;
	for (int i=1; i<=cinfo.totcells; i++)
		cellsStart[i] = cellsSize[i-1] + cellsStart[i-1];

	for (int i=0; i<np; i++)
	{
		const int cid = cinfo.getCellId(make_float3(coos[i].x[0], coos[i].x[1], coos[i].x[2]));
		buffer[cellsStart[cid]] = coos[i];
		cellsStart[cid]++;
	}

	for (int i=0; i<cinfo.totcells; i++)
		cellsStart[i] -= cellsSize[i];
}

void integrate(Particle* __restrict__ coos, const Force* __restrict__ accs, int np, float dt, float3 domainStart, float3 length)
{
	for (int i=0; i<np; i++)
	{
		coos[i].u[0] += accs[i].f[0]*dt;
		coos[i].u[1] += accs[i].f[1]*dt;
		coos[i].u[2] += accs[i].f[2]*dt;

		coos[i].x[0] += coos[i].u[0]*dt;
		coos[i].x[1] += coos[i].u[1]*dt;
		coos[i].x[2] += coos[i].u[2]*dt;

		if (coos[i].x[0] >  domainStart.x+length.x) coos[i].x[0] -= length.x;
		if (coos[i].x[0] <= domainStart.x)          coos[i].x[0] += length.x;

		if (coos[i].x[1] >  domainStart.y+length.y) coos[i].x[1] -= length.y;
		if (coos[i].x[1] <= domainStart.y)          coos[i].x[1] += length.y;

		if (coos[i].x[2] >  domainStart.z+length.z) coos[i].x[2] -= length.z;
		if (coos[i].x[2] <= domainStart.z)          coos[i].x[2] += length.z;
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

void forces(const Particle* __restrict__ coos, Force* __restrict__ accs, const int* __restrict__ cellsStart, const int* __restrict__ cellsSize,
		const CellListInfo& cinfo)
{

	const float ddt = 0.01;
	const float kBT = 1.0;
	const float gammadpd = 20;
	const float sigma = sqrt(2 * gammadpd * kBT);
	const float sigmaf = sigma / sqrt(ddt);
	const float aij = 50;

	const float3 length = cinfo.length;

	auto addForce = [=] (int dstId, int srcId, Force& a)
	{
		float _xr = coos[dstId].x[0] - coos[srcId].x[0];
		float _yr = coos[dstId].x[1] - coos[srcId].x[1];
		float _zr = coos[dstId].x[2] - coos[srcId].x[2];

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
				xr * (coos[dstId].u[0] - coos[srcId].u[0]) +
				yr * (coos[dstId].u[1] - coos[srcId].u[1]) +
				zr * (coos[dstId].u[2] - coos[srcId].u[2]);

		const float myrandnr = 0;//Logistic::mean0var1(1, min(srcId, dstId), max(srcId, dstId));

		const float strength = aij * argwr - (gammadpd * wr * rdotv + sigmaf * myrandnr) * wr;

		a.f[0] += strength * xr;
		a.f[1] += strength * yr;
		a.f[2] += strength * zr;

//		if (dstId == 8)
//			printf("%d -- %d  :  [%f %f %f %d] -- [%f %f %f %d] (dist2 %f) :  [%f %f %f]\n", dstId, srcId,
//					coos[dstId].x[0], coos[dstId].x[1], coos[dstId].x[2], coos[dstId].i1,
//					coos[srcId].x[0], coos[srcId].x[1], coos[srcId].x[2], coos[srcId].i1,
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

				for (int dstId = cellsStart[cid]; dstId < cellsStart[cid] + cellsSize[cid]; dstId++)
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

								for (int srcId = cellsStart[srcCid]; srcId < cellsStart[srcCid] + cellsSize[srcCid]; srcId++)
								{
									if (dstId != srcId)
									{
										float _xr = coos[dstId].x[0] - coos[srcId].x[0];
										float _yr = coos[dstId].x[1] - coos[srcId].x[1];
										float _zr = coos[dstId].x[2] - coos[srcId].x[2];

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
													xr * (coos[dstId].u[0] - coos[srcId].u[0]) +
													yr * (coos[dstId].u[1] - coos[srcId].u[1]) +
													zr * (coos[dstId].u[2] - coos[srcId].u[2]);

											const float myrandnr = 0;//Logistic::mean0var1(1, min(srcId, dstId), max(srcId, dstId));

											const float strength = aij * argwr - (gammadpd * wr * rdotv + sigmaf * myrandnr) * wr;

											a.f[0] += strength * xr;
											a.f[1] += strength * yr;
											a.f[2] += strength * zr;
										}
									}
								}
							}

					accs[dstId].f[0] = a.f[0];
					accs[dstId].f[1] = a.f[1];
					accs[dstId].f[2] = a.f[2];
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

	float3 length{16,16,16};
	float3 domainStart = -length / 2.0f;
	const float rc = 1.0f;
	ParticleVector dpds("dpd");
	CellList cells(&dpds, rc, domainStart, length);

	InitialConditions ic = createIC(config.child("node"));
	ic.exec(MPI_COMM_WORLD, &dpds, {0,0,0}, length);

	const float dt = 0.003;
	const float kBT = 1.0;
	const float gammadpd = 20;
	const float sigmadpd = sqrt(2 * gammadpd * kBT);
	const float sigma_dt = sigmadpd / sqrt(dt);
	const float adpd = 50;

	auto inter = [=] (ParticleVector* pv, CellList* cl, const float t, cudaStream_t stream) {
		interactionDPDSelf(pv, cl, t, stream, adpd, gammadpd, sigma_dt, rc);
	};

	auto haloInt = [=] (ParticleVector* pv1, ParticleVector* pv2, CellList* cl, const float t, cudaStream_t stream) {
		interactionDPDHalo(pv1, pv2, cl, t, stream, adpd, gammadpd, sigma_dt, rc);
	};

	dpds.coosvels.downloadFromDevice(true);
	dpds.forces.clear();
	int initialNP = dpds.np;

	HostBuffer<Particle> locparticles(dpds.np);
	for (int i=0; i<dpds.np; i++)
	{
		locparticles[i] = dpds.coosvels[i];

		locparticles[i].x[0] += (coords[0] + 0.5 - 0.5*ranks[0]) * length.x;
		locparticles[i].x[1] += (coords[1] + 0.5 - 0.5*ranks[1]) * length.z;
		locparticles[i].x[2] += (coords[2] + 0.5 - 0.5*ranks[2]) * length.y;
	}

	cudaStream_t defStream;
	CUDA_Check( cudaStreamCreateWithPriority(&defStream, cudaStreamNonBlocking, 10) );

	dpds.pushStreamWOhalo(defStream);

	HaloExchanger halo(cartComm);
	halo.attach(&dpds, &cells);
	Redistributor redist(cartComm);
	redist.attach(&dpds, &cells);

	const int niters = 3;

	printf("GPU execution\n");

	Timer tm;
	tm.start();

	for (int i=0; i<niters; i++)
	{
		cells.build(defStream);
		CUDA_Check( cudaStreamSynchronize(defStream) );

		dpds.forces.clear();
		inter(&dpds, &cells, dt*i, defStream);

		halo.exchange();
		haloInt(&dpds, &dpds, &cells, dt*i, defStream);

		integrateNoFlow(&dpds, dt, defStream);

		CUDA_Check( cudaStreamSynchronize(defStream) );

		redist.redistribute();
	}

	double elapsed = tm.elapsed() * 1e-9;

	printf("Finished in %f s, 1 step took %f ms\n", elapsed, elapsed / niters * 1000.0);

	if (argc < 5) return 0;
	cells.build(defStream);


	dpds.coosvels.downloadFromDevice(true);

	for (int i=0; i<dpds.np; i++)
	{
		dpds.coosvels[i].x[0] += (coords[0] + 0.5 - 0.5*ranks[0]) * length.x;
		dpds.coosvels[i].x[1] += (coords[1] + 0.5 - 0.5*ranks[1]) * length.z;
		dpds.coosvels[i].x[2] += (coords[2] + 0.5 - 0.5*ranks[2]) * length.y;
	}

	int totalParticles;
	int totalInitialPs;
	HostBuffer<int> sizes(ranks[0]*ranks[1]*ranks[2]), displs(ranks[0]*ranks[1]*ranks[2] + 1);
	HostBuffer<int> initialSizes(ranks[0]*ranks[1]*ranks[2]), initialDispls(ranks[0]*ranks[1]*ranks[2] + 1);

	MPI_Check( MPI_Gather(&dpds.np,   1, MPI_INT, sizes.hostPtr(),        1, MPI_INT, 0, MPI_COMM_WORLD) );
	MPI_Check( MPI_Gather(&initialNP, 1, MPI_INT, initialSizes.hostPtr(), 1, MPI_INT, 0, MPI_COMM_WORLD) );

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

	MPI_Check( MPI_Gatherv(dpds.coosvels.hostPtr(), dpds.np,   mpiPart, finalParticles.hostPtr(), sizes       .hostPtr(), displs       .hostPtr(), mpiPart, 0, MPI_COMM_WORLD) );
	MPI_Check( MPI_Gatherv(locparticles.hostPtr(),  initialNP, mpiPart, particles     .hostPtr(), initialSizes.hostPtr(), initialDispls.hostPtr(), mpiPart, 0, MPI_COMM_WORLD) );


	if (rank == 0)
	{
		float3 globDomainSize = make_float3(length.x*ranks[0], length.y*ranks[1], length.z*ranks[2]);
		CellListInfo globalCells(1.0f, -globDomainSize*0.5, globDomainSize);

		HostBuffer<Particle> buffer(totalInitialPs);
		HostBuffer<Force> accs(totalInitialPs);
		HostBuffer<int>   cellsStart(globalCells.totcells+1), cellsSize(globalCells.totcells+1);


		printf("CPU execution\n");
		printf("NP:  %d,  ref  %d\n", totalParticles, totalInitialPs);

		for (int i=0; i<niters; i++)
		{
			printf("%d...", i);
			fflush(stdout);
			makeCells(particles.hostPtr(), buffer.hostPtr(), cellsStart.hostPtr(), cellsSize.hostPtr(), totalInitialPs, globalCells);
			containerSwap(particles, buffer);

			forces(particles.hostPtr(), accs.hostPtr(), cellsStart.hostPtr(), cellsSize.hostPtr(), globalCells);
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
					finalParticles[i].x[0], finalParticles[i].x[1], finalParticles[i].x[2], finalParticles[i].i1);
				return 0;
			}

			if (0 <= particles[i].i1 && particles[i].i1 < totalInitialPs)
				cpuid[particles[i].i1] = i;
			else
			{
				printf("Wrong id on cpu: particle %d: [%f %f %f] %d\n", i,
					particles[i].x[0], particles[i].x[1], particles[i].x[2], particles[i].i1);
				return 0;
			}
		}


		double l2 = 0, linf = -1;

		for (int i=0; i<min(totalParticles, totalInitialPs); i++)
		{
			Particle cpuP = particles[cpuid[i]];
			Particle gpuP = finalParticles[gpuid[i]];

			double perr = -1;
			for (int c=0; c<3; c++)
			{
				const double err = fabs(cpuP.x[c] - gpuP.x[c]) + fabs(cpuP.u[c] - gpuP.u[c]);
				linf = max(linf, err);
				perr = max(perr, err);
				l2 += err * err;
			}

			if (argc > 5 && perr > 0.01)
			{
				printf("id %8d diff %8e  [%12f %12f %12f  %8d] [%12f %12f %12f] \n"
						"                           ref [%12f %12f %12f  %8d] [%12f %12f %12f] [%12f %12f %12f] \n\n", i, perr,
						gpuP.x[0], gpuP.x[1], gpuP.x[2], gpuP.i1,
						gpuP.u[0], gpuP.u[1], gpuP.u[2],
						cpuP.x[0], cpuP.x[1], cpuP.x[2], cpuP.i1,
						cpuP.u[0], cpuP.u[1], cpuP.u[2], accs[cpuid[i]].f[0], accs[cpuid[i]].f[1], accs[cpuid[i]].f[2]);
			}
		}

		l2 = sqrt(l2 / dpds.np);
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
