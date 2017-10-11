#include "vv.h"

#include <core/utils/kernel_launch.h>
#include <core/logger.h>
#include <core/pvs/particle_vector.h>

#include "integration_kernel.h"

#include "forcing_terms/none.h"
#include "forcing_terms/const_dp.h"
#include "forcing_terms/periodic_poiseuille.h"

template<class ForcingTerm>
void IntegratorVV<ForcingTerm>::stage1(ParticleVector* pv, float t, cudaStream_t stream)
{}

template<class ForcingTerm>
void IntegratorVV<ForcingTerm>::stage2(ParticleVector* pv, float t, cudaStream_t stream)
{
	static_assert(std::is_same<decltype(forcingTerm.setup(pv, t)), void>::value,
			"Forcing term functor must provide member"
			"void setup(ParticleVector*, float)");

	auto& _fterm = forcingTerm;
	_fterm.setup(pv, t);

	auto st2 = [_fterm] __device__ (Particle& p, const float3 f, const float invm, const float dt) {

		static_assert(std::is_same<decltype(_fterm(f, p)), float3>::value,
				"Forcing term functor must provide member"
				" __device__ float3 operator()(float3, Particle)");

		float3 modF = _fterm(f, p);

		p.u += modF*invm*dt;
		p.r += p.u*dt;
	};

	int nthreads = 128;
	debug2("Integrating (stage 2) %d %s particles, timestep is %f", pv->local()->size(), pv->name.c_str(), dt);

	auto pvView = create_PVview(pv, pv->local());
	SAFE_KERNEL_LAUNCH(
			integrationKernel,
			getNblocks(2*pvView.size, nthreads), nthreads, 0, stream,
			pvView, dt, st2 );

	pv->local()->changedStamp++;
}

template class IntegratorVV<Forcing_None>;
template class IntegratorVV<Forcing_ConstDP>;
template class IntegratorVV<Forcing_PeriodicPoiseuille>;
