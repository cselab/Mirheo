#include <core/celllist.h>
#include <core/containers.h>
#include <core/components.h>
#include <core/integrate.h>
#include <core/interactions.h>
#include <core/helper_math.h>
#include <core/wall.h>

#include <random>


	Integrator  createIntegrator(pugi::xml_node node)
	{
		Integrator result;
		result.name = node.attribute("name").as_string("");
		result.dt   = node.attribute("dt")  .as_float(0.01f);

		std::string type = node.attribute("type").as_string();

		if (type == "noflow")
		{
			result.integrate = integrateNoFlow;
		}

		if (type == "const_dp")
		{
			const float3 extraForce = node.attribute("extra_force").as_float3({0,0,0});

			result.integrate = [extraForce](ParticleVector* pv, const float dt, cudaStream_t stream) {
				integrateConstDP(pv, dt, stream, extraForce);
			};
		}

		return result;
	}

	Interaction createInteraction(pugi::xml_node node)
	{
		Interaction result;
		result.name = node.attribute("name").as_string("");

		result.rc = node.attribute("rc").as_float(1.0f);

		std::string type = node.attribute("type").as_string();

		if (type == "dpd")
		{
			const float rc = node.attribute("rc").as_float(1.0);
			const float dt = node.attribute("dt").as_float();
			const float kBT = node.attribute("kbt").as_float(1.0);
			const float gammadpd = node.attribute("gamma").as_float(20);
			const float sigmadpd = sqrt(2 * gammadpd * kBT);
			const float adpd = node.attribute("a").as_float(50);
			const float sigma_dt = sigmadpd / sqrt(dt);

			result.self = [=] (ParticleVector* pv, CellList* cl, const float t, cudaStream_t stream) {
				interactionDPDSelf(pv, cl, t, stream, adpd, gammadpd, sigma_dt, rc);
			};

			result.halo     = [=] (ParticleVector* pv1, ParticleVector* pv2, CellList* cl, const float t, cudaStream_t stream) {
				interactionDPDHalo(pv1, pv2, cl, t, stream, adpd, gammadpd, sigma_dt, rc);
			};

			result.external = [=] (ParticleVector* pv1, ParticleVector* pv2, CellList* cl, const float t, cudaStream_t stream) {
				interactionDPDExternal(pv1, pv2, cl, t, stream, adpd, gammadpd, sigma_dt, rc);
			};
		}

		return result;
	}


	InitialConditions createIC(pugi::xml_node node)
	{
		InitialConditions result;

		const float mass = node.attribute("mass")   .as_float(1.0);
		const float dens = node.attribute("density").as_float(1.0);

		result.exec = [=] (const MPI_Comm& comm, ParticleVector* pv, float3 globalDomainStart, float3 subDomainSize) {

			int3 ncells = make_int3( ceilf(subDomainSize) );
			float3 h = subDomainSize / make_float3(ncells);

			float volume = h.x*h.y*h.z;
			float avg = volume * dens;
			int predicted = round(avg * ncells.x*ncells.y*ncells.z * 1.05);
			pv->resize(predicted);

			int rank;
			MPI_Check( MPI_Comm_rank(comm, &rank) );

			const int seed = rank + 0;
			std::mt19937 gen(seed);
			std::poisson_distribution<> particleDistribution(avg);
			std::uniform_real_distribution<float> coordinateDistribution(0, 1);

			int mycount = 0;
			auto cooPtr = pv->coosvels.hostPtr();
			for (int i=0; i<ncells.x; i++)
				for (int j=0; j<ncells.y; j++)
					for (int k=0; k<ncells.z; k++)
					{
						int nparts = particleDistribution(gen);
						for (int p=0; p<nparts; p++)
						{
							pv->resize(mycount+1, resizePreserve);
							cooPtr[mycount].x[0] = i*h.x - 0.5*subDomainSize.x + coordinateDistribution(gen);
							cooPtr[mycount].x[1] = j*h.y - 0.5*subDomainSize.y + coordinateDistribution(gen);
							cooPtr[mycount].x[2] = k*h.z - 0.5*subDomainSize.z + coordinateDistribution(gen);
							cooPtr[mycount].i1 = mycount;

							cooPtr[mycount].u[0] = 0*coordinateDistribution(gen);
							cooPtr[mycount].u[1] = 0*coordinateDistribution(gen);
							cooPtr[mycount].u[2] = 0*coordinateDistribution(gen);

							cooPtr[mycount].i1 = mycount;
							mycount++;
						}
					}

			 pv->domainLength = subDomainSize;
			 pv->domainStart  = -subDomainSize*0.5;
			 pv->mass = mass;
			 pv->received = 0;

			 int totalCount=0; // TODO: int64!
			 MPI_Check( MPI_Exscan(&mycount, &totalCount, 1, MPI_INT, MPI_SUM, comm) );
			 for (int i=0; i < pv->np; i++)
				 cooPtr[i].i1 += totalCount;

			 pv->coosvels.uploadToDevice();

			 debug("Generated %d %s particles", pv->np, pv->name.c_str());
		};

		return result;
	}

	Wall createWall(pugi::xml_node node)
	{
		const float    createTm = node.attribute("creation_time").as_float(10.0);
		const std::string fname = node.attribute("file_name").as_string("sdf.dat");
		const float3          h = node.attribute("creation_time").as_float3({0.25, 0.25, 0.25});
		const std::string  name = node.attribute("name").as_string("");


		Wall wall(name, fname, h, createTm);
		return wall;
	}








