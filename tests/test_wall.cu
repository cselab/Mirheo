// Yo ho ho ho
#define private public

#include <core/containers.h>
#include <core/celllist.h>
#include <core/halo_exchanger.h>
#include <core/redistributor.h>
#include <core/logger.h>
#include <core/integrate.h>
#include <core/wall.h>

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

void integrate(Particle* __restrict__ coos, Force* __restrict__ accs, int np, float dt, float3 domainStart, float3 length)
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


void forces(const Particle* __restrict__ coos, Force* __restrict__ accs, const int* __restrict__ cellsStart, const int* __restrict__ cellsSize,
		int3 ncells, int totcells, float3 domainStart, float3 length)
{

	const float dt = 0.0025;
	const float kBT = 1.0;
	const float gammadpd = 20;
	const float sigma = sqrt(2 * gammadpd * kBT);
	const float sigmaf = sigma / sqrt(dt);
	const float aij = 50;

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
					Force a {0,0,0,0};

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

void createSdf(int3 resolution, float3 size, float r0, std::string fname)
{
	const float3 h = size / make_float3(resolution - 1);
	const float3 center = size / 2;
	float *sdf = new float[resolution.x * resolution.y * resolution.z];

	for (int i=0; i<resolution.z; i++)
	{
		for (int j=0; j<resolution.y; j++)
		{
			for (int k=0; k<resolution.x; k++)
			{
				float3 r = h * make_float3(i, j, k); // grid-centered data
				float3 dr = center - r;
				const float val = sqrtf(dr.x*dr.x + dr.y*dr.y + dr.z*dr.z) - r0;
				sdf[ (i*resolution.y + j)*resolution.x + k ] = val;

				//printf("%5.2f  ", val);
			}
			//printf("\n");
		}
		//printf("\n");
	}

	std::ofstream out(fname);
	out << size.x       << " " << size.y       << " " << size.z       << " " << std::endl;
	out << resolution.x << " " << resolution.y << " " << resolution.z << " " << std::endl;
	out.write((char*)sdf, resolution.x * resolution.y * resolution.z * sizeof(float));

	delete[] sdf;
}

void checkFrozenRemaining(Particle* frozen, int nFrozen, Particle* remaining, int nRem, Particle* initial, int n, float3 size, float r)
{
	std::vector<Particle> refFrozen, refRem;

	for (int i=0; i<n; i++)
	{
		const float sdf = sqrt(initial[i].x[0]*initial[i].x[0] + initial[i].x[1]*initial[i].x[1] + initial[i].x[2]*initial[i].x[2]) - r;
		if (sdf < 0.5f) refRem.push_back(initial[i]);
		if (-0.5f < sdf && sdf < 1.5f) refFrozen.push_back(initial[i]);
	}

	auto cmp = [](const Particle& a, const Particle& b) -> bool{
		float d1 = sqrt(a.x[0]*a.x[0] + a.x[1]*a.x[1] + a.x[2]*a.x[2]);
		float d2 = sqrt(b.x[0]*b.x[0] + b.x[1]*b.x[1] + b.x[2]*b.x[2]);

		return d1 < d2 && fabs(d1 - d2) > 1e-6;
	};

	std::sort(refFrozen.begin(), refFrozen.end(), cmp);
	std::sort(refRem.begin(), refRem.end(), cmp);

	std::sort(frozen, frozen + nFrozen, cmp);
	std::sort(remaining, remaining + nRem, cmp);

	std::vector<Particle> res(n);
	auto vecend = std::set_intersection(frozen, frozen + nFrozen, remaining, remaining + nRem, res.begin(), cmp);
	if (vecend - res.begin())
	{
		printf("Whoops, %d  frozen and remaining particles are the same!\n", res.size());
	}

	vecend = std::set_difference(frozen, frozen + nFrozen, refFrozen.begin(), refFrozen.end(), res.begin(), cmp);
	for (auto p=res.begin(); p!=vecend; p++)
	{
		printf("Missing particle in Frozen: [%f %f %f]\n", p->x[0], p->x[1], p->x[2]);
	}

	vecend = std::set_difference(remaining, remaining + nRem, refRem.begin(), refRem.end(), res.begin(), cmp);
	for (auto p=res.begin(); p!=vecend; p++)
	{
		printf("Missing particle in Remaining: [%f %f %f] \n", p->x[0], p->x[1], p->x[2]);
	}

	//======================================

	vecend = std::set_difference(refFrozen.begin(), refFrozen.end(), frozen, frozen + nFrozen, res.begin(), cmp);
	float maxdiff = 0;
	for (auto p=res.begin(); p!=vecend; p++)
	{
		float sdf = sqrt(p->x[0]*p->x[0] + p->x[1]*p->x[1] + p->x[2]*p->x[2]) - r;
		maxdiff = max(maxdiff, min(sdf, 1-sdf));

		//printf("haha: [%f %f %f]  %f\n", p->x[0], p->x[1], p->x[2], sdf);

	}
	printf("Max distance inside frozen layer of missed particles: %f\n", maxdiff);


	vecend = std::set_difference(refRem.begin(), refRem.end(), remaining, remaining + nRem, res.begin(), cmp);
	maxdiff = 0;
	for (auto p=res.begin(); p!=vecend; p++)
	{
		float sdf = sqrt(p->x[0]*p->x[0] + p->x[1]*p->x[1] + p->x[2]*p->x[2]) - r;
		maxdiff = min(maxdiff, sdf);

		//printf("hohoho: [%f %f %f]  %f\n", p->x[0], p->x[1], p->x[2], sdf);

	}
	printf("Min sdf of missed remaining particles: %f\n", maxdiff);
}

int main(int argc, char ** argv)
{
	// Init

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

	float3 length{32,32,32};
	float3 domainStart = -length / 2.0f;
	const float rc = 1.0f;
	ParticleVector dpds("dpd");
	CellList cells(&dpds, rc, domainStart, length);

	InitialConditions ic = createIC(config.child("node"));
	ic.exec(MPI_COMM_WORLD, &dpds, {0,0,0}, length);

	const float radius = 10;
	createSdf(make_int3(129), make_float3(ncells), radius, "sphere.sdf");

	int c = 0;
	for (int i=0; i<dpds.np; i++)
	{
		dpds.coosvels[i].u[0] = 20*(drand48() - 0.5);
		dpds.coosvels[i].u[1] = 20*(drand48() - 0.5);
		dpds.coosvels[i].u[2] = 20*(drand48() - 0.5);
	}
	dpds.coosvels.uploadToDevice();

	HostBuffer<Particle> initial(dpds.np);
	memcpy(initial.hostPtr(), dpds.coosvels.hostPtr(), dpds.np*sizeof(Particle));


	dpds.forces.clear();

	cudaStream_t defStream;
	CUDA_Check( cudaStreamCreateWithPriority(&defStream, cudaStreamNonBlocking, 10) );

	HaloExchanger halo(cartComm);
	halo.attach(&dpds, ndens);
	Redistributor redist(cartComm, config);
	redist.attach(&dpds, ndens);

	cells.build(defStream);
	CUDA_Check( cudaStreamSynchronize(defStream) );

	const float dt = config.getFloat("Common", "dt");
	printf("%f\n", dt);
	const int niters = 50;

	Wall wall("wall", "sphere.sdf", {1, 1, 1}, -1);
	wall.create(comm, domainStart, length, length, &dpds, &cells);
	cells.build(defStream);
	wall.attach(&dpds);

	HostBuffer<Particle> frozen(wall.frozen.size);
	HostBuffer<float> intSdf(wall.sdfRawData.size);

	intSdf.copy(wall.sdfRawData);
	frozen.copy(wall.frozen.coosvels);

//	printf("============================================================================================\n");
//	for (int i=0; i<wall.resolution.z; i++)
//	{
//		for (int j=0; j<wall.resolution.y; j++)
//		{
//			for (int k=0; k<wall.resolution.x; k++)
//			{
//				printf("%5.2f  ", intSdf[ (i*wall.resolution.y + j)*wall.resolution.x + k ]);
//			}
//			printf("\n");
//		}
//		printf("\n");
//	}

	if (argc > 1)
		checkFrozenRemaining(frozen.hostPtr(), frozen.size, dpds.coosvels.hostPtr(), dpds.np, initial.hostPtr(), initial.size, make_float3(ncells), radius);

	integrateNoFlow(&dpds, dt, defStream);
	wall.bounce(defStream);
	dpds.coosvels.downloadFromDevice();



//	for (int i=0; i<niters; i++)
//	{
//		printf("Iteration %d\n", i);
//		dpds.accs.clear(defStream);
//		computeInternalDPD(dpds, defStream);
//
//		halo.exchangeInit();
//		halo.exchangeFinalize();
//
//		computeHaloDPD(dpds, defStream);
//		CUDA_Check( cudaStreamSynchronize(defStream) );
//
//		wall.computeInteractions(defStream);
//		wall.bounce(defStream);
//
//		redist.redistribute(dt);
//
//		buildCellListAndIntegrate(dpds, config, dt, defStream);
//		CUDA_Check( cudaStreamSynchronize(defStream) );
//		wall._check();
//	}

	if (argc < 2) return 0;

	int np = particles.size;
	int totcells = dpds.totcells;

	HostBuffer<Particle> buffer(np);
	HostBuffer<Force> accs(np);
	HostBuffer<int>   cellsStart(totcells+1), cellsSize(totcells+1);

	printf("CPU execution\n");

	for (int i=0; i<niters; i++)
	{
		printf("%d...", i);
		fflush(stdout);
		makeCells(particles.hostPtr(), buffer.hostPtr(), cellsStart.hostPtr(), cellsSize.hostPtr(), np, ncells, totcells, domainStart, 1.0f);
		forces(particles.hostPtr(), accs.hostPtr(), cellsStart.hostPtr(), cellsSize.hostPtr(), ncells, totcells, domainStart, length);
		integrate(particles.hostPtr(), accs.hostPtr(), np, dt, domainStart, length);
	}

	printf("\nDone, checking\n");
	printf("NP:  %d,  ref  %d\n", dpds.np, np);


	dpds.coosvels.synchronize(synchronizeHost);

	std::vector<int> gpuid(np), cpuid(np);
	for (int i=0; i<np; i++)
	{
		gpuid[dpds.coosvels[i].i1] = i;
		cpuid[particles[i].i1] = i;
	}


	double l2 = 0, linf = -1;

	for (int i=0; i<np; i++)
	{
		Particle cpuP = particles[cpuid[i]];
		Particle gpuP = dpds.coosvels[gpuid[i]];

		double perr = -1;
		for (int c=0; c<3; c++)
		{
			const double err = fabs(cpuP.x[c] - gpuP.x[c]) + fabs(cpuP.u[c] - gpuP.u[c]);
			linf = max(linf, err);
			perr = max(perr, err);
			l2 += err * err;
		}

		if (argc > 2 && perr > 0.01)
		{
			printf("id %8d diff %8e  [%12f %12f %12f  %8d] [%12f %12f %12f]\n"
				   "                           ref [%12f %12f %12f  %8d] [%12f %12f %12f] \n\n", i, perr,
					gpuP.x[0], gpuP.x[1], gpuP.x[2], gpuP.i1,
					gpuP.u[0], gpuP.u[1], gpuP.u[2],
					cpuP.x[0], cpuP.x[1], cpuP.x[2], cpuP.i1,
					cpuP.u[0], cpuP.u[1], cpuP.u[2]);
		}
	}

	l2 = sqrt(l2 / dpds.np);
	printf("L2   norm: %f\n", l2);
	printf("Linf norm: %f\n", linf);

	return 0;
}
