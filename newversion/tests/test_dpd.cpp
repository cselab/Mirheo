
#include "../core/containers.h"
#include "../core/celllist.h"
#include "../core/dpd.h"
#include "../core/halo_exchanger.h"
#include "../core/logger.h"

#include <unistd.h>

Logger logger;

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

	logger.init(MPI_COMM_WORLD, "onerank.log", 9);

	MPI_Check( MPI_Comm_size(MPI_COMM_WORLD, &nranks) );
	MPI_Check( MPI_Comm_rank(MPI_COMM_WORLD, &rank) );
	MPI_Check( MPI_Cart_create(MPI_COMM_WORLD, 3, ranks, periods, 0, &cartComm) );

	// Initial cells

	int3 ncells = {96, 96, 96};
	float3 domainStart = {-ncells.x / 2.0f, -ncells.y / 2.0f, -ncells.z / 2.0f};
	float3 length{(float)ncells.x, (float)ncells.y, (float)ncells.z};
	ParticleVector dpds(ncells, domainStart, length);

	const int ndens = 4;
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

					dpds.coosvels[c].u[0] = drand48() - 0.5;
					dpds.coosvels[c].u[1] = drand48() - 0.5;
					dpds.coosvels[c].u[2] = drand48() - 0.5;
					c++;
				}

	dpds.resize(c);
	dpds.coosvels.synchronize(synchronizeDevice);

	cudaStream_t defStream;
	CUDA_Check( cudaStreamCreateWithPriority(&defStream, cudaStreamNonBlocking, 10) );

	HaloExchanger halo(cartComm);
	halo.attach(&dpds, 7);

	buildCellList((float4*)dpds.coosvels.devdata, dpds.np, dpds.domainStart, dpds.ncells, 1.0f, (float4*)dpds.pingPongBuf.devdata, dpds.cellsSize.devdata, dpds.cellsStart.devdata, defStream);
	swap(dpds.coosvels, dpds.pingPongBuf, defStream);
	CUDA_Check( cudaStreamSynchronize(defStream) );

	for (int i=0; i<10; i++)
	{
		//halo.exchangeInit();

		//cudaDeviceSynchronize();

		computeInternalDPD(dpds, defStream);
		//computeHaloDPD(dpds, defStream);
		//buildCellList((float4*)dpds.coosvels.devdata, dpds.np, dpds.domainStart, dpds.ncells, 1.0f, (float4*)dpds.pingPongBuf.devdata, dpds.cellsSize.devdata, dpds.cellsStart.devdata, defStream);
		//swap(dpds.coosvels, dpds.pingPongBuf, defStream);

		//halo.exchangeFinalize();

		//cudaDeviceSynchronize();

		//cudaStreamSynchronize(defStream);
	}

	const float dt = 0.0025;
	const float kBT = 1.0;
	const float gammadpd = 20;
	const float sigma = sqrt(2 * gammadpd * kBT);
	const float sigmaf = 126.4911064;//sigma / sqrt(dt);
	const float aij = 50;


	HostBuffer<Particle> hacc, particles;
	hacc.copy(dpds.accs);
	particles.copy(dpds.coosvels);

	cudaDeviceSynchronize();

	printf("finished, reducing acc\n");
	double a[3] = {};
	for (int i=0; i<np; i++)
	{
		for (int c=0; c<3; c++)
			a[c] += hacc[i].a[c];
	}
	printf("Reduced acc: %e %e %e\n\n", a[0], a[1], a[2]);


	printf("Checking (this is not necessarily a cubic domain)......\n");
	std::vector<Acceleration> refAcc(acc.size);

	auto addForce = [&](int dstId, int srcId, Acceleration& a)
	{
		const float _xr = particles[dstId].x[0] - particles[srcId].x[0];
		const float _yr = particles[dstId].x[1] - particles[srcId].x[1];
		const float _zr = particles[dstId].x[2] - particles[srcId].x[2];

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
				xr * (particles[dstId].u[0] - particles[srcId].u[0]) +
				yr * (particles[dstId].u[1] - particles[srcId].u[1]) +
				zr * (particles[dstId].u[2] - particles[srcId].u[2]);

		const float myrandnr = 0;//Logistic::mean0var1(1, min(srcId, dstId), max(srcId, dstId));

		const float strength = aij * argwr - (gammadpd * wr * rdotv + sigmaf * myrandnr) * wr;

		a.a[0] += strength * xr;
		a.a[1] += strength * yr;
		a.a[2] += strength * zr;
	};

#pragma omp parallel for collapse(3)
	for (int cx = 0; cx < lx; cx++)
		for (int cy = 0; cy < ly; cy++)
			for (int cz = 0; cz < lz; cz++)
			{
				const int cid = (cz*ly + cy)*lx + cx;

				for (int dstId = hcellsstart[cid]; dstId < hcellsstart[cid+1]; dstId++)
				{
					Acceleration a {0,0,0};

					for (int dx = -1; dx <= 1; dx++)
						for (int dy = -1; dy <= 1; dy++)
							for (int dz = -1; dz <= 1; dz++)
							{
								const int srcCid = ( (cz+dz)*ly + (cy+dy) ) * lx + cx+dx;
								if (srcCid >= ncells || srcCid < 0) continue;

								for (int srcId = hcellsstart[srcCid]; srcId < hcellsstart[srcCid+1]; srcId++)
								{
									if (dstId != srcId)
										addForce(dstId, srcId, a);
								}
							}

					refAcc[dstId].a[0] = a.a[0];
					refAcc[dstId].a[1] = a.a[1];
					refAcc[dstId].a[2] = a.a[2];
				}
			}


	double l2 = 0, linf = -1;

	for (int i=0; i<np; i++)
	{
		double perr = -1;
		for (int c=0; c<3; c++)
		{
			const double err = fabs(refAcc[i].a[c] - hacc[i].a[c]);
			linf = max(linf, err);
			perr = max(perr, err);
			l2 += err * err;
		}

		if (argc > 1 && perr > 0.1)
		{
			printf("id %d,  %12f %12f %12f     ref %12f %12f %12f    diff   %12f %12f %12f\n", i,
				hacc[i].a[0], hacc[i].a[1], hacc[i].a[2],
				refAcc[i].a[0], refAcc[i].a[1], refAcc[i].a[2],
				hacc[i].a[0]-refAcc[i].a[0], hacc[i].a[1]-refAcc[i].a[1], hacc[i].a[2]-refAcc[i].a[2]);
		}
	}


	l2 = sqrt(l2 / np);
	printf("L2   norm: %f\n", l2);
	printf("Linf norm: %f\n", linf);

	CUDA_Check( cudaPeekAtLastError() );
	return 0;
}
