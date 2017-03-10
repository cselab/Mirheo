
#include <core/particle_vector.h>
#include <core/celllist.h>
#include <core/halo_exchanger.h>
#include <core/logger.h>
#include <core/components.h>
#include <core/interactions.h>

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

	std::string xml = R"(<node mass="1.0" density="8.0">)";
	pugi::xml_document config;
	config.load_string(xml.c_str());

	float3 length{64,64,64};
	float3 domainStart = -length / 2.0f;
	const float rc = 1.0f;
	ParticleVector dpds("dpd");
	CellList cells(&dpds, rc, domainStart, length);

	InitialConditions ic = createIC(config.child("node"));
	ic.exec(MPI_COMM_WORLD, &dpds, {0,0,0}, length);

	const int np = dpds.np;
	HostBuffer<Particle> initial(np);
	auto initPtr = initial.hostPtr();
	for (int i=0; i<np; i++)
		initPtr[i] = dpds.coosvels[i];

	cells.build(0);

	const float dt = 0.0025;
	const float kBT = 1.0;
	const float gammadpd = 20;
	const float sigmadpd = sqrt(2 * gammadpd * kBT);
	const float sigma_dt = sigmadpd / sqrt(dt);
	const float adpd = 50;

	auto inter = [=] (ParticleVector* pv, CellList* cl, const float t, cudaStream_t stream) {
		interactionDPDSelf(pv, cl, t, stream, adpd, gammadpd, sigma_dt, rc);
	};

	for (int i=0; i<100; i++)
	{
		dpds.forces.clear();
		inter(&dpds, &cells, 0, 0);

		cudaDeviceSynchronize();
	}

	dpds.coosvels.downloadFromDevice(false);
	dpds.forces.downloadFromDevice(true);



	HostBuffer<Force> hacc;
	HostBuffer<int> hcellsstart;
	HostBuffer<uint8_t> hcellssize;
	hcellsstart.copy(cells.cellsStart, 0);
	hcellssize.copy(cells.cellsSize, 0);
	hacc.copy(dpds.forces);

	cudaDeviceSynchronize();

	printf("finished, reducing acc\n");
	double a[3] = {};
	for (int i=0; i<dpds.np; i++)
	{
		for (int c=0; c<3; c++)
			a[c] += hacc[i].f[c];
	}
	printf("Reduced acc: %e %e %e\n\n", a[0], a[1], a[2]);


	printf("Checking (this is not necessarily a cubic domain)......\n");

	std::vector<Force> refAcc(hacc.size());

	auto addForce = [&](int dstId, int srcId, Force& a)
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

		const float strength = adpd * argwr - (gammadpd * wr * rdotv + sigma_dt * myrandnr) * wr;

		a.f[0] += strength * xr;
		a.f[1] += strength * yr;
		a.f[2] += strength * zr;
	};

#pragma omp parallel for collapse(3)
	for (int cx = 0; cx < cells.ncells.x; cx++)
		for (int cy = 0; cy < cells.ncells.y; cy++)
			for (int cz = 0; cz < cells.ncells.z; cz++)
			{
				const int cid = cells.encode(cx, cy, cz);

				const int2 start_size = cells.decodeStartSize(hcellsstart[cid]);

				for (int dstId = start_size.x; dstId < start_size.x + start_size.y; dstId++)
				{
					Force a {0,0,0,0};

					for (int dx = -1; dx <= 1; dx++)
						for (int dy = -1; dy <= 1; dy++)
							for (int dz = -1; dz <= 1; dz++)
							{
								const int srcCid = cells.encode(cx+dx, cy+dy, cz+dz);
								if (srcCid >= cells.totcells || srcCid < 0) continue;

								const int2 srcStart_size = cells.decodeStartSize(hcellsstart[srcCid]);

								for (int srcId = srcStart_size.x; srcId < srcStart_size.x + srcStart_size.y; srcId++)
								{
									if (dstId != srcId)
										addForce(dstId, srcId, a);
								}
							}

					refAcc[dstId].f[0] = a.f[0];
					refAcc[dstId].f[1] = a.f[1];
					refAcc[dstId].f[2] = a.f[2];
				}
			}


	double l2 = 0, linf = -1;

	for (int i=0; i<dpds.np; i++)
	{
		double perr = -1;

		double toterr = 0;
		for (int c=0; c<3; c++)
		{
			const double err = fabs(refAcc[i].f[c] - hacc[i].f[c]);
			toterr += err;
			linf = max(linf, err);
			perr = max(perr, err);
			l2 += err * err;
		}

		if (argc > 1 && (perr > 0.1 || std::isnan(toterr)))

		{
			printf("id %d,  %12f %12f %12f     ref %12f %12f %12f    diff   %12f %12f %12f\n", i,
				hacc[i].f[0], hacc[i].f[1], hacc[i].f[2],
				refAcc[i].f[0], refAcc[i].f[1], refAcc[i].f[2],
				hacc[i].f[0]-refAcc[i].f[0], hacc[i].f[1]-refAcc[i].f[1], hacc[i].f[2]-refAcc[i].f[2]);
		}
	}


	l2 = sqrt(l2 / dpds.np);
	printf("L2   norm: %f\n", l2);
	printf("Linf norm: %f\n", linf);

	CUDA_Check( cudaPeekAtLastError() );
	return 0;
}
