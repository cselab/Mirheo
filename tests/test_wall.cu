// Yo ho ho ho
#define private public

#include <core/particle_vector.h>
#include <core/celllist.h>
#include <core/mpi/api.h>
#include <core/containers.h>
#include <core/logger.h>
#include <core/integrate.h>
#include <core/wall.h>
#include <core/xml/pugixml.hpp>

#include "timer.h"
#include <unistd.h>

Logger logger;

void makeCells(Particle*& __restrict__ coos, Particle*& __restrict__ buffer, int* __restrict__ cellsStartSize, int* __restrict__ cellsSize,
		int np, CellListInfo cinfo)
{
	for (int i=0; i<cinfo.totcells+1; i++)
		cellsSize[i] = 0;

	for (int i=0; i<np; i++)
		cellsSize[cinfo.getCellId(float3{coos[i].x[0], coos[i].x[1], coos[i].x[2]})]++;

	cellsStartSize[0] = 0;
	for (int i=1; i<=cinfo.totcells; i++)
		cellsStartSize[i] = cellsSize[i-1] + cellsStartSize[i-1];

	for (int i=0; i<np; i++)
	{
		const int cid = cinfo.getCellId(float3{coos[i].x[0], coos[i].x[1], coos[i].x[2]});
		buffer[cellsStartSize[cid]] = coos[i];
		cellsStartSize[cid]++;
	}

	for (int i=0; i<cinfo.totcells; i++)
		cellsStartSize[i] -= cellsSize[i];

	std::swap(coos, buffer);
}

void integrate(Particle* __restrict__ coos, Force* __restrict__ accs, int np, float dt, CellListInfo cinfo)
{
	for (int i=0; i<np; i++)
	{
		coos[i].u[0] += accs[i].f[0]*dt;
		coos[i].u[1] += accs[i].f[1]*dt;
		coos[i].u[2] += accs[i].f[2]*dt;

		coos[i].x[0] += coos[i].u[0]*dt;
		coos[i].x[1] += coos[i].u[1]*dt;
		coos[i].x[2] += coos[i].u[2]*dt;

		if (coos[i].x[0] >   0.5f * cinfo.domainSize.x) coos[i].x[0] -= cinfo.domainSize.x;
		if (coos[i].x[0] <= -0.5f * cinfo.domainSize.x)	coos[i].x[0] += cinfo.domainSize.x;

		if (coos[i].x[1] >   0.5f * cinfo.domainSize.y) coos[i].x[1] -= cinfo.domainSize.y;
		if (coos[i].x[1] <= -0.5f * cinfo.domainSize.x)	coos[i].x[1] += cinfo.domainSize.y;

		if (coos[i].x[2] >   0.5f * cinfo.domainSize.z) coos[i].x[2] -= cinfo.domainSize.z;
		if (coos[i].x[2] <= -0.5f * cinfo.domainSize.x)	coos[i].x[2] += cinfo.domainSize.z;
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


void forces(const Particle* __restrict__ coos, Force* __restrict__ accs, const int* __restrict__ cellsStartSize, const int* __restrict__ cellsSize,
		CellListInfo cinfo)
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

		_xr = minabs(_xr, _xr - cinfo.domainSize.x, _xr + cinfo.domainSize.x);
		_yr = minabs(_yr, _yr - cinfo.domainSize.y, _yr + cinfo.domainSize.y);
		_zr = minabs(_zr, _zr - cinfo.domainSize.z, _zr + cinfo.domainSize.z);

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
	};

	const int3 ncells = cinfo.ncells;

#pragma omp parallel for collapse(3)
	for (int cx = 0; cx < ncells.x; cx++)
		for (int cy = 0; cy < ncells.y; cy++)
			for (int cz = 0; cz < ncells.z; cz++)
			{
				const int cid = cinfo.encode(cx, cy, cz);

				for (int dstId = cellsStartSize[cid]; dstId < cellsStartSize[cid] + cellsSize[cid]; dstId++)
				{
					Force f {0,0,0,0};

					for (int dx = -1; dx <= 1; dx++)
						for (int dy = -1; dy <= 1; dy++)
							for (int dz = -1; dz <= 1; dz++)
							{
								int ncx, ncy, ncz;
								ncx = (cx+dx + ncells.x) % ncells.x;
								ncy = (cy+dy + ncells.y) % ncells.y;
								ncz = (cz+dz + ncells.z) % ncells.z;

								const int srcCid = cinfo.encode(ncx, ncy, ncz);
								if (srcCid >= cinfo.totcells || srcCid < 0) continue;

								for (int srcId = cellsStartSize[srcCid]; srcId < cellsStartSize[srcCid] + cellsSize[srcCid]; srcId++)
								{
									if (dstId != srcId)
										addForce(dstId, srcId, f);

									//printf("%d  %f %f %f\n", dstId, a.a[0], a.a[1], a.a[2]);
								}
							}

					accs[dstId].f[0] = f.f[0];
					accs[dstId].f[1] = f.f[1];
					accs[dstId].f[2] = f.f[2];
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


float solve2_01(float a, float b, float c)
{
	if (fabs(a) < 1e-6)
	{
		if (fabs(b) < 1e-6) return -1;

		const float t = -c/b;
		if (0 <= t && t <= 1) return t;
		else return -1;
	}

	const float D = b*b - 4*a*c;
	if (D < 0) return -1;

	const float sqrtD = sqrt(D);
	const float t1 = 0.5*(-b + sqrtD) / a;
	const float t2 = 0.5*(-b - sqrtD) / a;

	if ( 0 <= t1 && t1 <= 1 && 0 <= t2 && t2 <= 1 ) return min(t1, t2);
	else if (0 <= t1 && t1 <= 1) return t1;
	else if (0 <= t2 && t2 <= 1) return t2;
	else return -1;
}

float bounce(float r, float3 x0, float3 x1)
{
	const float a = dot(x0 - x1, x0 - x1);
	const float b = 2 * dot(x0, x1-x0);
	const float c = dot(x0, x0) - r*r;

	return solve2_01(a, b, c);
}


void bounceAll(Particle* coosvels, int n, const float r, const float dt)
{
	for (int i=0; i<n; i++)
	{
		const float3 coo = {coosvels[i].x[0], coosvels[i].x[1], coosvels[i].x[2]};
		const float3 vel = {coosvels[i].u[0], coosvels[i].u[1], coosvels[i].u[2]};

		const float3 coo0 = coo - vel*dt;

		const float t = bounce(r, coo0, coo);
		if (t > -0.5f)
		{
			const float3 newcoo = coo0 + t * vel*dt + (1-t) * (vel*(-dt));
			coosvels[i].x[0] = newcoo.x;
			coosvels[i].x[1] = newcoo.y;
			coosvels[i].x[2] = newcoo.z;
		}
	}
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

	logger.init(MPI_COMM_WORLD, "wall.log", 9);

	MPI_Check( MPI_Comm_size(MPI_COMM_WORLD, &nranks) );
	MPI_Check( MPI_Comm_rank(MPI_COMM_WORLD, &rank) );
	MPI_Check( MPI_Cart_create(MPI_COMM_WORLD, 3, ranks, periods, 0, &cartComm) );


	std::string xml = R"(<node mass="1.0" density="2.0">)";
	pugi::xml_document config;
	config.load_string(xml.c_str());

	float3 length{32, 32, 32};
	float3 domainStart = -length / 2.0f;
	const float rc = 1.0f;
	ParticleVector dpds("dpd");
	CellList cells(&dpds, rc, length);

	InitialConditions ic = createIC(config.child("node"));
	ic.exec(MPI_COMM_WORLD, &dpds, {0,0,0}, length);

	const float radius = 5;
	auto evalSdf = [radius] (float x, float y, float z) {
		return sqrt(x*x + y*y + z*z) - radius;
	};

	createSdf(make_int3(189), make_float3(cells.ncells), radius, "sphere.sdf");

	int c = 0;
	for (int i=0; i<dpds.local()->size(); i++)
	{
		dpds.local()->coosvels[i].u[0] = 5*(drand48() - 0.5);
		dpds.local()->coosvels[i].u[1] = 5*(drand48() - 0.5);
		dpds.local()->coosvels[i].u[2] = 5*(drand48() - 0.5);
	}
	dpds.local()->coosvels.uploadToDevice();

	HostBuffer<Particle> initial(dpds.local()->size());
	memcpy(initial.hostPtr(), dpds.local()->coosvels.hostPtr(), dpds.local()->size()*sizeof(Particle));

	dpd.local()->forces.clear();

	cudaStream_t defStream;
	CUDA_Check( cudaStreamCreateWithPriority(&defStream, cudaStreamNonBlocking, 10) );
	dpds.pushStream(defStream);

	HaloExchanger halo(cartComm, defStream);
	halo->attach(&dpds, &cells);
	Redistributor redist(cartComm);
	redist.attach(&dpds, &cells);

	cells.build(defStream);
	CUDA_Check( cudaStreamSynchronize(defStream) );

	const float dt = 0.1;
	const int niters = 50;

	Wall wall("wall", "sphere.sdf", {1/3.0, 1/3.0, 1/3.0}, -1);
	wall.create(cartComm, {0,0,0}, length, length, &dpds, &cells);
	cells.build(defStream);
	wall.attach(&dpds, &cells);

	HostBuffer<Particle> frozen(wall.frozen.size());
	HostBuffer<float> intSdf(wall.sdfRawData.size());

	intSdf.copy(wall.sdfRawData, defStream);
	frozen.copy(wall.frozen.local()->coosvels, defStream);
	dpds.local()->coosvels.downloadFromDevice();

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
		checkFrozenRemaining(frozen.hostPtr(), frozen.size(), dpds.local()->coosvels.hostPtr(), dpds.local()->size(), initial.hostPtr(), initial.size(), make_float3(cells.ncells), radius);

	integrateNoFlow(&dpds, dt, defStream);

	dpds.local()->coosvels.downloadFromDevice();
	HostBuffer<Particle> particles(dpds.local()->size());
	memcpy(particles.hostPtr(), dpds.local()->coosvels.hostPtr(), dpds.local()->size()*sizeof(Particle));

	wall.bounce(dt, defStream);
	dpds.local()->coosvels.downloadFromDevice();

	bounceAll(particles.hostPtr(), particles.size(), radius, dt);

	printf("CPU bounce finished\n");


//	for (int i=0; i<niters; i++)
//	{
//		printf("Iteration %d\n", i);
//		dpds.accs.clear(defStream);
//		computeInternalDPD(dpds, defStream);
//
//		halo->exchangeInit();
//		halo->exchangeFinalize();
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
//
//	if (argc < 2) return 0;
//
//	int np = particles.size;
//	int totcells = dpds.totcells;
//
//	HostBuffer<Particle> buffer(np);
//	HostBuffer<Force> accs(np);
//	HostBuffer<int>   cellsStartSize(totcells+1), cellsSize(totcells+1);
//
//	printf("CPU execution\n");
//
//	for (int i=0; i<niters; i++)
//	{
//		printf("%d...", i);
//		fflush(stdout);
//		makeCells(particles.hostPtr(), buffer.hostPtr(), cellsStartSize.hostPtr(), cellsSize.hostPtr(), np, ncells, totcells, domainStart, 1.0f);
//		forces(particles.hostPtr(), accs.hostPtr(), cellsStartSize.hostPtr(), cellsSize.hostPtr(), ncells, totcells, domainStart, length);
//		integrate(particles.hostPtr(), accs.hostPtr(), np, dt, domainStart, length);
//	}
//
//	printf("\nDone, checking\n");
//	printf("NP:  %d,  ref  %d\n", dpds.local()->size(), np);
//
//
//	dpds.local()->coosvels.synchronize(synchronizeHost);

	int np = dpds.local()->size();
//	std::vector<int> gpuid(np), cpuid(np);
//	for (int i=0; i<np; i++)
//	{
//		gpuid[dpds.local()->coosvels[i].i1] = i;
//		cpuid[particles[i].i1] = i;
//	}

	double l2 = 0, linf = -1;

	for (int i=0; i<np; i++)
	{
		Particle cpuP = particles[i];
		Particle gpuP = dpds.local()->coosvels[i];

		double perr = -1;
		for (int c=0; c<3; c++)
		{
			const double err = fabs(cpuP.x[c] - gpuP.x[c]);// + fabs(cpuP.u[c] - gpuP.u[c]);
			linf = max(linf, err);
			perr = max(perr, err);
			l2 += err * err;
		}

		if (argc > 2 && (perr > 0.01 || evalSdf(cpuP.x[0], cpuP.x[1], cpuP.x[2]) > 0))
		{
			printf("id %8d diff %8e  [%12f %12f %12f  %8d] (%f)\n"
				   "                           ref [%12f %12f %12f  %8d] (%f) \n\n", i, perr,
					gpuP.x[0], gpuP.x[1], gpuP.x[2], gpuP.i1, evalSdf(gpuP.x[0], gpuP.x[1], gpuP.x[2]),
					cpuP.x[0], cpuP.x[1], cpuP.x[2], cpuP.i1, evalSdf(cpuP.x[0], cpuP.x[1], cpuP.x[2]) );
		}
	}

	l2 = sqrt(l2 / dpds.local()->size());
	printf("L2   norm: %f\n", l2);
	printf("Linf norm: %f\n", linf);

	return 0;
}
