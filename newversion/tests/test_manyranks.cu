// Yo ho ho ho
#define private public

#include "../core/containers.h"
#include "../core/celllist.h"
#include "../core/dpd.h"
#include "../core/halo_exchanger.h"
#include "../core/redistributor.h"
#include "../core/logger.h"
#include "../core/integrate.h"

#include "timer.h"
#include <unistd.h>

Logger logger;

void makeCells(Particle*& __restrict__ coos, Particle*& __restrict__ buffer, int* __restrict__ cellsStart, int* __restrict__ cellsSize,
		int np, int3 ncells, int totcells, float3 domainStart, float invrc)
{
	for (int i=0; i<totcells+1; i++)
		cellsSize[i] = 0;

	for (int i=0; i<np; i++)
		cellsSize[getCellId(float3{coos[i].x[0], coos[i].x[1], coos[i].x[2]}, domainStart, ncells, invrc)]++;

	cellsStart[0] = 0;
	for (int i=1; i<=totcells; i++)
		cellsStart[i] = cellsSize[i-1] + cellsStart[i-1];

	for (int i=0; i<np; i++)
	{
		const int cid = getCellId(float3{coos[i].x[0], coos[i].x[1], coos[i].x[2]}, domainStart, ncells, invrc);
		buffer[cellsStart[cid]] = coos[i];
		cellsStart[cid]++;
	}

	for (int i=0; i<totcells; i++)
		cellsStart[i] -= cellsSize[i];

	std::swap(coos, buffer);
}

void integrate(Particle* __restrict__ coos, Acceleration* __restrict__ accs, int np, float dt, float3 domainStart, float3 length)
{
	for (int i=0; i<np; i++)
	{
		coos[i].u[0] += accs[i].a[0]*dt;
		coos[i].u[1] += accs[i].a[1]*dt;
		coos[i].u[2] += accs[i].a[2]*dt;

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
T minabs(T arg)
{
	return arg;
}

template<typename T, typename... Args>
T minabs(T arg, Args... other)
{
	const T v = minabs(other...	);
	return (std::abs(arg) < std::abs(v)) ? arg : v;
}


void forces(const Particle* __restrict__ coos, Acceleration* __restrict__ accs, const int* __restrict__ cellsStart, const int* __restrict__ cellsSize,
		int3 ncells, int totcells, float3 domainStart, float3 length)
{

	const float ddt = 0.0025;
	const float kBT = 1.0;
	const float gammadpd = 20;
	const float sigma = sqrt(2 * gammadpd * kBT);
	const float sigmaf = sigma / sqrt(ddt);
	const float aij = 50;

	auto addForce = [=] (int dstId, int srcId, Acceleration& a)
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

		a.a[0] += strength * xr;
		a.a[1] += strength * yr;
		a.a[2] += strength * zr;
	};

#pragma omp parallel for collapse(3)
	for (int cx = 0; cx < ncells.x; cx++)
		for (int cy = 0; cy < ncells.y; cy++)
			for (int cz = 0; cz < ncells.z; cz++)
			{
				const int cid = encode(cx, cy, cz, ncells);

				for (int dstId = cellsStart[cid]; dstId < cellsStart[cid] + cellsSize[cid]; dstId++)
				{
					Acceleration a {0,0,0,0};

					for (int dx = -1; dx <= 1; dx++)
						for (int dy = -1; dy <= 1; dy++)
							for (int dz = -1; dz <= 1; dz++)
							{
								int ncx, ncy, ncz;
								ncx = (cx+dx + ncells.x) % ncells.x;
								ncy = (cy+dy + ncells.y) % ncells.y;
								ncz = (cz+dz + ncells.z) % ncells.z;

								const int srcCid = encode(ncx, ncy, ncz, ncells);
								if (srcCid >= totcells || srcCid < 0) continue;

								for (int srcId = cellsStart[srcCid]; srcId < cellsStart[srcCid] + cellsSize[srcCid]; srcId++)
								{
									if (dstId != srcId)
										addForce(dstId, srcId, a);

									//printf("%d  %f %f %f\n", dstId, a.a[0], a.a[1], a.a[2]);
								}
							}

					accs[dstId].a[0] = a.a[0];
					accs[dstId].a[1] = a.a[1];
					accs[dstId].a[2] = a.a[2];
				}
			}
}

int main(int argc, char ** argv)
{
	// Init

	int provided;
	MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
	if (provided < MPI_THREAD_MULTIPLE)
	{
	    printf("ERROR: The MPI library does not have full thread support\n");
	    MPI_Abort(MPI_COMM_WORLD, 1);
	}

	logger.init(MPI_COMM_WORLD, "manyranks.log", 0);

	if (argc != 4)
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


	// Initial cells

	int3 ncells = {32, 32, 32};
	float3 domainStart = {-ncells.x / 2.0f, -ncells.y / 2.0f, -ncells.z / 2.0f};
	float3 length{(float)ncells.x, (float)ncells.y, (float)ncells.z};
	ParticleVector dpds(ncells, domainStart, length);

	const int ndens = 8;
	dpds.resize(ncells.x*ncells.y*ncells.z * ndens);

	srand48(rank);

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
					dpds.coosvels[c].i1 = c + rank * dpds.np;

					dpds.coosvels[c].u[0] = 2*(drand48() - 0.5);
					dpds.coosvels[c].u[1] = 2*(drand48() - 0.5);
					dpds.coosvels[c].u[2] = 2*(drand48() - 0.5);
					c++;
				}


	const int initialNP = c;
	dpds.resize(c);
	dpds.coosvels.synchronize(synchronizeDevice);
	dpds.accs.clear();

	HostBuffer<Particle> locparticles(dpds.np);
	for (int i=0; i<dpds.np; i++)
	{
		locparticles[i] = dpds.coosvels[i];

		locparticles[i].x[0] += (coords[0] + 0.5) * length.x;
		locparticles[i].x[1] += (coords[1] + 0.5) * length.z;
		locparticles[i].x[2] += (coords[2] + 0.5) * length.y;
	}

	cudaStream_t defStream;
	CUDA_Check( cudaStreamCreateWithPriority(&defStream, cudaStreamNonBlocking, 10) );

	HaloExchanger halo(cartComm);
	halo.attach(&dpds, ndens);
	Redistributor redist(cartComm);
	redist.attach(&dpds, ndens);

	buildCellList(dpds,defStream);
	CUDA_Check( cudaStreamSynchronize(defStream) );

	const float dt = 0.002;
	const int niters = 200;

	printf("GPU execution\n");

	Timer tm;
	tm.start();

	for (int i=0; i<niters; i++)
	{
		dpds.accs.clear(defStream);
		computeInternalDPD(dpds, defStream);

		halo.exchangeInit();
		halo.exchangeFinalize();

		computeHaloDPD(dpds, defStream);
		CUDA_Check( cudaStreamSynchronize(defStream) );

		redist.redistribute(dt);

		buildCellListAndIntegrate(dpds, dt, defStream);
		CUDA_Check( cudaStreamSynchronize(defStream) );
	}

	double elapsed = tm.elapsed() * 1e-9;

	printf("Finished in %f s, 1 step took %f ms\n", elapsed, elapsed / niters * 1000.0);

	return 0;

	dpds.coosvels.synchronize(synchronizeHost);

	for (int i=0; i<dpds.np; i++)
	{
		dpds.coosvels[i].x[0] += (coords[0] + 0.5) * length.x;
		dpds.coosvels[i].x[1] += (coords[1] + 0.5) * length.z;
		dpds.coosvels[i].x[2] += (coords[2] + 0.5) * length.y;
	}

	int totparts;
	HostBuffer<int> sizes(ranks[0]*ranks[1]*ranks[2]), displs(ranks[0]*ranks[1]*ranks[2] + 1);
	MPI_Check( MPI_Gather(&dpds.np, 1, MPI_INT, sizes.hostdata, 1, MPI_INT, 0, MPI_COMM_WORLD) );

	displs[0] = 0;
	for (int i=1; i<ranks[0]*ranks[1]*ranks[2]+1; i++)
	{
		displs[i] = displs[i-1]+sizes[i-1];

		if (rank == 0) printf("%d %d %d \n", i-1, sizes[i-1], displs[i]);
	}
	totparts = max(displs[displs.size - 1], 0);

	HostBuffer<Particle> all      (totparts);
	HostBuffer<Particle> particles( ranks[0]*ranks[1]*ranks[2] * initialNP );

	MPI_Datatype mpiPart;
	MPI_Check( MPI_Type_contiguous(sizeof(Particle), MPI_BYTE, &mpiPart) );
	MPI_Check( MPI_Type_commit(&mpiPart) );

	MPI_Check( MPI_Gatherv(dpds.coosvels.hostdata, dpds.np, mpiPart, all.hostdata, sizes.hostdata, displs.hostdata, mpiPart, 0, MPI_COMM_WORLD) );
	MPI_Check( MPI_Gather(locparticles.hostdata,  initialNP, mpiPart, particles.hostdata, initialNP, mpiPart, 0, MPI_COMM_WORLD) );


	if (rank == 0)
	{
		int np = all.size;

		ncells *= make_int3  (ranks[0], ranks[1], ranks[2]);
		length *= make_float3(ranks[0], ranks[1], ranks[2]);
		domainStart = make_float3(0, 0, 0);

		int maxdim = std::max({ncells.x, ncells.y, ncells.z});
		int minpow2 = 1;
		while (minpow2 < maxdim) minpow2 *= 2;
		int totcells = minpow2*minpow2*minpow2;

		HostBuffer<Particle> buffer(np);
		HostBuffer<Acceleration> accs(np);
		HostBuffer<int>   cellsStart(totcells+1), cellsSize(totcells+1);

		printf("CPU execution\n");

		for (int i=0; i<niters; i++)
		{
			printf("%d...", i);
			fflush(stdout);
			makeCells(particles.hostdata, buffer.hostdata, cellsStart.hostdata, cellsSize.hostdata, np, ncells, totcells, domainStart, 1.0f);
			forces(particles.hostdata, accs.hostdata, cellsStart.hostdata, cellsSize.hostdata, ncells, totcells, domainStart, length);
			integrate(particles.hostdata, accs.hostdata, np, dt, domainStart, length);
		}

		printf("\nDone, checking\n");
		printf("NP:  %d,  ref  %d\n", totparts, particles.size);


		dpds.coosvels.synchronize(synchronizeHost);

		std::vector<int> gpuid(np), cpuid(np);
		for (int i=0; i<np; i++)
		{
			gpuid[all[i].i1] = i;
			cpuid[particles[i].i1] = i;
		}


		double l2 = 0, linf = -1;

		for (int i=0; i<totparts; i++)
		{
			Particle cpuP = particles[cpuid[i]];
			Particle gpuP = all[gpuid[i]];

			double perr = -1;
			for (int c=0; c<3; c++)
			{
				const double err = fabs(cpuP.x[c] - gpuP.x[c]) + fabs(cpuP.u[c] - gpuP.u[c]);
				linf = max(linf, err);
				perr = max(perr, err);
				l2 += err * err;
			}

			if (argc > 1 && perr > 0.01)
			{
				printf("id %8d diff %8e  [%12f %12f %12f  %8d] [%12f %12f %12f]\n"
						"                           ref [%12f %12f %12f  %8d] [%12f %12f %12f] [%12f %12f %12f] \n\n", i, perr,
						gpuP.x[0], gpuP.x[1], gpuP.x[2], gpuP.i1,
						gpuP.u[0], gpuP.u[1], gpuP.u[2],
						cpuP.x[0], cpuP.x[1], cpuP.x[2], cpuP.i1,
						cpuP.u[0], cpuP.u[1], cpuP.u[2], accs[cpuid[i]].a[0], accs[cpuid[i]].a[1], accs[cpuid[i]].a[2]);
			}
		}

		l2 = sqrt(l2 / dpds.np);
		printf("L2   norm: %f\n", l2);
		printf("Linf norm: %f\n", linf);
	}

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
