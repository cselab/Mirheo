
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

	int provided;
	MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
	if (provided < MPI_THREAD_MULTIPLE)
	{
	    printf("ERROR: The MPI library does not have full thread support\n");
	    MPI_Abort(MPI_COMM_WORLD, 1);
	}

	logger.init(MPI_COMM_WORLD, "onerank.log", 9);

	// Initial cells

	int l = 64;
	int3 ncells = {l, l, l};
	float3 domainStart = {-ncells.x / 2.0f, -ncells.y / 2.0f, -ncells.z / 2.0f};
	float3 length{(float)ncells.x, (float)ncells.y, (float)ncells.z};
	ParticleVector dpds(ncells, domainStart, length);

	const int ndens = 8;
	dpds.resize(ncells.x*ncells.y*ncells.z * ndens);

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

//	HaloExchanger halo(cartComm);
//	halo.attach(&dpds, 7);

	buildCellList((float4*)dpds.coosvels.devdata, dpds.np, dpds.domainStart, dpds.ncells, dpds.totcells, 1.0f, (float4*)dpds.pingPongBuf.devdata, dpds.cellsSize.devdata, dpds.cellsStart.devdata, defStream);
	swap(dpds.coosvels, dpds.pingPongBuf, defStream);
	CUDA_Check( cudaStreamSynchronize(defStream) );

	for (int i=0; i<50; i++)
	{
		//halo.exchangeInit();

		//cudaDeviceSynchronize();

		computeInternalDPD(dpds, defStream);
		//computeHaloDPD(dpds, defStream);
		//buildCellList((float4*)dpds.coosvels.devdata, dpds.np, dpds.domainStart, dpds.ncells, 1.0f, (float4*)dpds.pingPongBuf.devdata, dpds.cellsSize.devdata, dpds.cellsStart.devdata, defStream);
		//swap(dpds.coosvels, dpds.pingPongBuf, defStream);

		//halo.exchangeFinalize();

		cudaDeviceSynchronize();

		//cudaStreamSynchronize(defStream);
	}

	const float dt = 0.0025;
	const float kBT = 1.0;
	const float gammadpd = 20;
	const float sigma = sqrt(2 * gammadpd * kBT);
	const float sigmaf = sigma / sqrt(dt);
	const float aij = 50;


	HostBuffer<Acceleration> hacc;
	HostBuffer<int> hcellsstart;
	HostBuffer<uint8_t> hcellssize;
	hcellsstart.copy(dpds.cellsStart);
	hcellssize.copy(dpds.cellsSize);
	hacc.copy(dpds.accs);

	dpds.coosvels.synchronize(synchronizeHost);
	cudaDeviceSynchronize();

	printf("finished, reducing acc\n");
	double a[3] = {};
	for (int i=0; i<dpds.np; i++)
	{
		for (int c=0; c<3; c++)
			a[c] += hacc[i].a[c];
	}
	printf("Reduced acc: %e %e %e\n\n", a[0], a[1], a[2]);


	printf("Checking (this is not necessarily a cubic domain)......\n");
	return 0;

	std::vector<Acceleration> refAcc(hacc.size);

	auto addForce = [&](int dstId, int srcId, Acceleration& a)
	{
		const float _xr = dpds.coosvels[dstId].x[0] - dpds.coosvels[srcId].x[0];
		const float _yr = dpds.coosvels[dstId].x[1] - dpds.coosvels[srcId].x[1];
		const float _zr = dpds.coosvels[dstId].x[2] - dpds.coosvels[srcId].x[2];

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
				xr * (dpds.coosvels[dstId].u[0] - dpds.coosvels[srcId].u[0]) +
				yr * (dpds.coosvels[dstId].u[1] - dpds.coosvels[srcId].u[1]) +
				zr * (dpds.coosvels[dstId].u[2] - dpds.coosvels[srcId].u[2]);

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

				const int2 start_size = decodeStartSize(hcellsstart[cid]);

				for (int dstId = start_size.x; dstId < start_size.x + start_size.y; dstId++)
				{
					Acceleration a {0,0,0,0};

					for (int dx = -1; dx <= 1; dx++)
						for (int dy = -1; dy <= 1; dy++)
							for (int dz = -1; dz <= 1; dz++)
							{
								const int srcCid = encode(cx+dx, cy+dy, cz+dz, ncells);
								if (srcCid >= dpds.totcells || srcCid < 0) continue;

								const int2 srcStart_size = decodeStartSize(hcellsstart[srcCid]);

								for (int srcId = srcStart_size.x; srcId < srcStart_size.x + srcStart_size.y; srcId++)
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

	for (int i=0; i<dpds.np; i++)
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


	l2 = sqrt(l2 / dpds.np);
	printf("L2   norm: %f\n", l2);
	printf("Linf norm: %f\n", linf);

	CUDA_Check( cudaPeekAtLastError() );
	return 0;
}
