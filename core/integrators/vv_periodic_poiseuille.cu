#include "vv_periodic_poiseuille.h"
#include "integration_kernel.h"

#include <core/logger.h>
#include <core/pvs/particle_vector.h>


void IntegratorVVPeriodicPoiseuille::stage1(ParticleVector* pv, cudaStream_t stream)
{ }

void IntegratorVVPeriodicPoiseuille::stage2(ParticleVector* pv, cudaStream_t stream)
{
	auto pvView = create_PVview(pv, pv->local());

	// Workaround for debug
	int _dir;
	switch (dir)
	{
		case Direction::x: _dir = 0; break;
		case Direction::y: _dir = 1; break;
		case Direction::z: _dir = 2; break;
	}

	auto _globalDomainSize = globalDomainSize;
	auto _force = force;

	auto st2 = [_dir, _globalDomainSize, _force, pvView] __device__ (Particle& p, const float3 f, const float invm, const float dt) {
		float3 gr = pvView.local2global(p.r);
		float3 ef{0.0f,0.0f,0.0f};

		if (_dir == 0)
			ef.x = gr.y > 0.5f*_globalDomainSize.y ? _force : -_force;

		if (_dir == 1)
			ef.y = gr.z > 0.5f*_globalDomainSize.z ? _force : -_force;

		if (_dir == 2)
			ef.z = gr.x > 0.5f*_globalDomainSize.x ? _force : -_force;

		p.u += (f+ef)*invm*dt;
		p.r += p.u*dt;
	};

	int nthreads = 128;
	debug2("Integrating (stage 2) %d %s particles with periodic poiseuille force %f, timestep is %f",
			force, pv->local()->size(), pv->name.c_str(), dt);

	if (pv->local()->size() > 0)
		integrationKernel<<< getNblocks(2*pvView.size, nthreads), nthreads, 0, stream >>>(pvView, dt, st2);

	pv->local()->changedStamp++;
}
