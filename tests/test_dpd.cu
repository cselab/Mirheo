#define private public

#include <core/particle_vector.h>
#include <core/celllist.h>
#include <core/mpi/api.h>
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
	CellList cells(&dpds, rc, length);
	cells.makePrimary();

	InitialConditions ic = createIC(config.child("node"));
	ic.exec(MPI_COMM_WORLD, &dpds, {0,0,0}, length);

	const int np = dpds.local()->size();
	HostBuffer<Particle> initial(np);
	auto initPtr = initial.hostPtr();
	for (int i=0; i<np; i++)
		initPtr[i] = dpds.local()->coosvels[i];

	cells.setStream(0);
	cells.build();

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

	for (int i=0; i<1; i++)
	{
		dpds.local()->forces.clear();
		inter(&dpds, &cells, 0, 0);

		cudaDeviceSynchronize();
	}

	dpds.local()->coosvels.downloadFromDevice();

	HostBuffer<Force> hacc;
	HostBuffer<uint> hcellsstart;
	HostBuffer<uint8_t> hcellssize;
	hcellsstart.copy(cells.cellsStartSize, 0);
	hcellssize.copy(cells.cellsSize, 0);
	hacc.copy(dpds.local()->forces, 0);

	cudaDeviceSynchronize();

	printf("finished, reducing acc\n");
	double3 a = {};
	for (int i=0; i<dpds.local()->size(); i++)
	{
		a.x += hacc[i].f.x;
		a.y += hacc[i].f.y;
		a.z += hacc[i].f.z;
	}
	printf("Reduced acc: %e %e %e\n\n", a.x, a.y, a.z);


	printf("Checking (this is not necessarily a cubic domain)......\n");

	std::vector<Force> refAcc(hacc.size());

	auto addForce = [&](int dstId, int srcId, Force& a)
	{
		const float _xr = dpds.local()->coosvels[dstId].r.x - dpds.local()->coosvels[srcId].r.x;
		const float _yr = dpds.local()->coosvels[dstId].r.y - dpds.local()->coosvels[srcId].r.y;
		const float _zr = dpds.local()->coosvels[dstId].r.z - dpds.local()->coosvels[srcId].r.z;

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
				xr * (dpds.local()->coosvels[dstId].u.x - dpds.local()->coosvels[srcId].u.x) +
				yr * (dpds.local()->coosvels[dstId].u.y - dpds.local()->coosvels[srcId].u.y) +
				zr * (dpds.local()->coosvels[dstId].u.z - dpds.local()->coosvels[srcId].u.z);

		const float myrandnr = 0;//Logistic::mean0var1(1, min(srcId, dstId), max(srcId, dstId));

		const float strength = adpd * argwr - (gammadpd * wr * rdotv + sigma_dt * myrandnr) * wr;

		a.f.x += strength * xr;
		a.f.y += strength * yr;
		a.f.z += strength * zr;
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

					refAcc[dstId].f.x = a.f.x;
					refAcc[dstId].f.y = a.f.y;
					refAcc[dstId].f.z = a.f.z;
				}
			}


	double l2 = 0, linf = -1;

	for (int i=0; i<dpds.local()->size(); i++)
	{
		double perr = -1;

		double toterr = 0;
		for (int c=0; c<3; c++)
		{
			double err;
			if (c==0) err = fabs(refAcc[i].f.x - hacc[i].f.x);
			if (c==1) err = fabs(refAcc[i].f.y - hacc[i].f.y);
			if (c==2) err = fabs(refAcc[i].f.z - hacc[i].f.z);

			toterr += err;
			linf = max(linf, err);
			perr = max(perr, err);
			l2 += err * err;
		}

		if (argc > 1 && (perr > 0.1 || std::isnan(toterr)))

		{
			printf("id %d,  %12f %12f %12f     ref %12f %12f %12f    diff   %12f %12f %12f\n", i,
				hacc[i].f.x, hacc[i].f.y, hacc[i].f.z,
				refAcc[i].f.x, refAcc[i].f.y, refAcc[i].f.z,
				hacc[i].f.x-refAcc[i].f.x, hacc[i].f.y-refAcc[i].f.y, hacc[i].f.z-refAcc[i].f.z);
		}
	}


	l2 = sqrt(l2 / dpds.local()->size());
	printf("L2   norm: %f\n", l2);
	printf("Linf norm: %f\n", linf);

	CUDA_Check( cudaPeekAtLastError() );
	return 0;
}
