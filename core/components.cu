#include "components.h"
#include "integrate.h"

namespace uDeviceX
{
	Integrator  createIntegrator(std::string name, pugi::xml_node node)
	{
		Integrator result;
		result.name = name;

		std::string type = node.attribute("type").as_string();

		if (type == "noflow")
		{
			result.integrate = [](ParticleVector* pv, const float dt, cudaStream_t stream) {
				auto noflow = [] __device__ (float4& x, float4& v, const float4 f, const float invm, const float dt, const int pid) {
					_noflow(x, v, f, invm, dt);
				};

				integrationKernel<<< (2*pv.np + 127)/128, 128, 0, stream >>>((float4*)pv.coosvels.devPtr(), (float4*)pv.forces.constDevPtr(), pv.np, dt, noflow);
			};
		}

		if (type == "constDP")
		{
			const float3 extraForce = node.attribute("extra_force").as_float3({0,0,0});

			result.integrate = [=](ParticleVector* pv, const float dt, cudaStream_t stream) {
				auto noflow = [extraForce] __device__ (float4& x, float4& v, const float4 f, const float invm, const float dt, const int pid) {
					_constDP(x, v, f, invm, dt, extraForce);
				};

				integrationKernel<<< (2*pv.np + 127)/128, 128, 0, stream >>>((float4*)pv.coosvels.devPtr(), (float4*)pv.forces.constDevPtr(), pv.np, dt, constDP);
			};
		}

		return result;
	}


	Interaction createInteraction(std::string name, pugi::xml_node node)
	{
		Interaction result;
		result.name = name;

		rc = node.attribute("rc").as_float(1.0f);

		std::string type = node.attribute("type").as_string();

		if (type == "dpd")
		{
			const float dt = node.attribute("dt").as_float();
			const float kBT = node.attribute("kbt").as_float(1.0);
			const float gammadpd = node.attribute("gamma").as_float(20);
			const float sigmadpd = sqrt(2 * gammadpd * kBT);
			const float adpd = node.attribute("a").as_float(50);
			const float sigma_dt = sigmadpd / sqrt(dt);

			result.self = [=] (ParticleVector* pv, CellList* cl, const float time, cudaStream_t stream) {
				auto dpdInt = [=] __device__ ( const float3 dstCoo, const float3 dstVel, const int dstId,
								   const float3 srcCoo, const float3 srcVel, const int srcId)
				{
					return dpd_interaction(dstCoo, dstVel, dstId, srcCoo, srcVel, srcId, adpd, gammadpd, sigma_dt, seed);
				};

				const int nth = 32 * 4;

				if (pv->np > 0)
				{
					debug("Computing internal forces for %s (%d particles)", pv.name, pv.size());
					computeSelfInteractions<<< (pv->np + nth - 1) / nth, nth, 0, stream >>>(
							(float4*)pv->coosvels.constDevPtr(), (float*)pv->forces.devPtr(), cl->cellInfo(), cl->cellsStart.constDevPtr(), pv->np, dpdInt);
				}
			};

			result.halo = [=] (ParticleVector* pv1, ParticleVector* pv2, CellList* cl, const float time, cudaStream_t stream) {
				auto dpdInt = [=] __device__ ( const float3 dstCoo, const float3 dstVel, const int dstId,
								   const float3 srcCoo, const float3 srcVel, const int srcId)
				{
					return dpd_interaction(dstCoo, dstVel, dstId, srcCoo, srcVel, srcId, adpd, gammadpd, sigma_dt, seed);
				};

				const int nth = 32 * 4;

				if (pv1->np > 0 && pv2->np > 0)
				{
					debug("Computing halo forces for %s with %d halo particles", pv->name, pv.halo.size());
					computeExternalInteractions<false, true> <<< (pv2->halo.size() + nth - 1) / nth, nth, 0, stream >>>(
								(float4*)pv2->halo.constDevPtr(), nullptr, (float4*)pv1->coosvels.constDevPtr(),
								(float*)pv1->forces.devPtr(), cl->cellInfo(), cl->cellsStart.constDevPtr(), pv2->halo.size(), dpdInt);
				}
			};

			result.external = [=] (ParticleVector* pv1, ParticleVector* pv2, CellList* cl, const float time, cudaStream_t stream) {
				auto dpdInt = [=] __device__ ( const float3 dstCoo, const float3 dstVel, const int dstId,
								   const float3 srcCoo, const float3 srcVel, const int srcId)
				{
					return dpd_interaction(dstCoo, dstVel, dstId, srcCoo, srcVel, srcId, adpd, gammadpd, sigma_dt, seed);
				};

				const int nth = 32 * 4;

				if (pv1->np > 0 && pv2->np > 0)
				{
					debug("Computing external forces between %s (%d) and %s (%d) for %d ext paricles", pv.halo.size);
							computeExternalInteractions<true, true> <<< (pv2->np + nth - 1) / nth, nth, 0, stream >>>(
										(float4*)pv2->coosvels.constDevPtr(), nullptr, (float4*)pv1->coosvels.constDevPtr(),
										(float*)pv1->forces.devPtr(), cl->cellInfo(), cl->cellsStart.constDevPtr(), pv2->np, dpdInt);
				}
			};
		}

		return result;
	}

}
