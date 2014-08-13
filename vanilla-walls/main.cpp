#include <cstdlib>
#include <cmath>
#include <cstdio>
#include <cassert>

#include <vector>
#include <string>
#include <iostream>

#ifdef USE_CUDA
#include "cuda-dpd.h"
#endif
#include "funnel-obstacle.h"

#include "particles.h"
#include "funnel-bouncer.h"

//******** SEM integration *****************
#ifdef USE_SEM
#include "cell-factory.h"
#include "cuda-sem.h"

struct ParticlesSEM : Particles
{
    ParticlesSEM(const int n, const float Lx)
    : Particles(n, Lx)
    {}

    ParamsSEM params;

    void internal_forces(const float kBT, const double dt)
	{
	    //abort();

	    for(int i = 0; i < n; ++i)
	    {
		xa[i] = xg;
		ya[i] = yg;
		za[i] = zg;
	    }

	    
	    /* internal_forces_bipartite(kBT, dt,
				  &xp.front(), &yp.front(), &zp.front(),
				  &xv.front(), &yv.front(), &zv.front(), n, myidstart, myidstart);*/
	   forces_sem_cuda_direct( &xp.front(), &yp.front(), &zp.front(),
				    &xv.front(), &yv.front(), &zv.front(),
				    &xa.front(), &ya.front(), &za.front(),
				    n, params.rcutoff, L[0], L[1], L[2], params.gamma, kBT, dt, params.u0, params.rho, params.req, params.D, params.rc);
	}
};

void interforces(Particles& dpdp, ParticlesSEM& semp, double kBT, double dt)
{
    const float xdomainsize = dpdp.L[0];
    const float ydomainsize = dpdp.L[1];
    const float zdomainsize = dpdp.L[2];

    const float gamma = 45;
    const float sigma = sqrt(2 * gamma * kBT);
    const float aij = 2.5;
    forces_dpd_cuda_bipartite(&dpdp.xp.front(), &dpdp.yp.front(), &dpdp.zp.front(),
			      &dpdp.xv.front(), &dpdp.yv.front(), &dpdp.zv.front(),
			      &dpdp.xa.front(), &dpdp.ya.front(), &dpdp.za.front(),
			      dpdp.n, dpdp.myidstart,
			      &semp.xp.front(), &semp.yp.front(), &semp.zp.front(),
			      &semp.xv.front(), &semp.yv.front(), &semp.zv.front(),
			      &semp.xa.front(), &semp.ya.front(), &semp.za.front(),
			      semp.n, semp.myidstart,
			      1,
			      xdomainsize,
			      ydomainsize,
			      zdomainsize,
			      aij,
			      gamma,
			      sigma,  1.0f / sqrt(dt));
}

void microfluidics_simulation(Particles& dpdp, ParticlesSEM& semp, const float kBT,
        const double tend, const double dt, const Bouncer* bouncer)
{
    dpdp.internal_forces(kBT, dt);
    semp.internal_forces(kBT, dt);

    dpdp.vmd_xyz("mf-ic-dpd.xyz", false);
    semp.vmd_xyz("mf-ic-dpd.xyz", false);

    const size_t nt = (int)(tend / dt);

    for(int it = 0; it < nt; ++it)
    {
        dpdp.update_stage1(dt, it);
        semp.update_stage1(dt, it);

        dpdp.internal_forces(kBT, dt);
        semp.internal_forces(kBT, dt);

        if (bouncer != nullptr) {
            bouncer->bounce(dpdp, dt);
            bouncer->bounce(semp, dt);
        }

        interforces(dpdp, semp, kBT, dt);
        if (bouncer != nullptr) {
            bouncer->compute_forces(kBT, dt, dpdp);
            bouncer->compute_forces(kBT, dt, semp);
        }

        dpdp.update_stage2(dt, it);
        semp.update_stage2(dt, it);
    }
}

void runSemWithObstacles(const float L, const int Nm, const int n, const float dt, 
    const Bouncer& bouncer, const Particles& remaining)
{
    const int ncellpts = 1000;
    vector<float> cellpts(ncellpts * 3);
    ParamsSEM semp;
    cell_factory(ncellpts, &cellpts.front(), semp);

    double com[3] = {0, 0, 0}, rmax = 0;
    {
    for(int i = 0; i < ncellpts; ++i)
        for(int c =0; c < 3; ++c)
        com[c] += cellpts[c + 3 * i];

    for(int c =0 ; c< 3; ++c)
        com[c] /= ncellpts;

        for(int i = 0; i < ncellpts; ++i)
    {
        double r2 = 0;
        for(int c =0; c < 3; ++c)
        r2 += pow(cellpts[c + 3 * i] - com[c], 2);

        rmax = max(rmax, sqrt(r2));
    }

    double goal[3] = { -3, -10, 0};

    for(int i = 0; i < ncellpts; ++i)
        for(int c =0; c < 3; ++c)
        cellpts[c + 3 * i] += goal[c] - com[c];

    printf("COM: %f %f %f R %f\n", com[0], com[1], com[2], rmax);

    com[0] = com[1] = com[2] = 0;

    for(int i = 0; i < ncellpts; ++i)
        for(int c =0; c < 3; ++c)
        com[c] += cellpts[c + 3 * i];

    for(int c =0 ; c< 3; ++c)
        com[c] /= ncellpts;

    printf("COM: %f %f %f R %f\n", com[0], com[1], com[2], rmax);

    //exit(0);
    }

    ParticlesSEM cell(ncellpts, L);
    cell.params = semp;
    for(int i = 0; i < ncellpts; ++i)
    {
    cell.xp[i] = cellpts[0 + 3 * i];
    cell.yp[i] = cellpts[1 + 3 * i];
    cell.zp[i] = cellpts[2 + 3 * i];
    }
    cell.name = "cell";
    cell.vmd_xyz("cell.xyz");
    cell.yg = 0.00;
    cell.steps_per_dump = 5;
    //cell.equilibrate(.1, 30*3*5, 0.02);
    //exit(0);


    Particles sys(0, L);
    sys.name = "solvent";
    sys.yg = 0.01;
    sys.steps_per_dump = 5;

    {
    //vector<bool> mark(remainin.n);

    for(int i = 0; i < remaining.n; ++i)
    {
        const double r2 =
        pow(remaining.xp[i] - com[0], 2) +
        pow(remaining.yp[i] - com[1], 2) +
        pow(remaining.zp[i] - com[2], 2) ;

        if (r2 >= rmax * rmax)
        {
        sys.xp.push_back(remaining.xp[i]);
        sys.yp.push_back(remaining.yp[i]);
        sys.zp.push_back(remaining.zp[i]);

        sys.xv.push_back(remaining.xv[i]);
        sys.yv.push_back(remaining.yv[i]);
        sys.zv.push_back(remaining.zv[i]);

        sys.xa.push_back(0);
        sys.ya.push_back(0);
        sys.za.push_back(0);

        sys.n++;
        }
    }

    sys.acquire_global_id();
    }

     microfluidics_simulation(sys, cell, .1, 30*3*5, 0.02, &bouncer);
    //sys.equilibrate(.1, 30*3*5, 0.02);
}
#endif

//******************************************

int main()
{
    const float L = 40;
    const int Nm = 3;
    const int n = L * L * L * Nm;
    const float dt = 0.02;

    Particles particles(n, L);
    particles.equilibrate(0.1, 100 * dt, dt, nullptr);

    const float sandwich_half_width = L / 2 - 1.7;
#if 1
    TomatoSandwich bouncer(L);
    bouncer.radius2 = 4;
    bouncer.half_width = sandwich_half_width;
#else
    SandwichBouncer bouncer(L);
    bouncer.half_width = sandwich_half_width;
#endif
    Particles remaining = bouncer.carve(particles);
    bouncer.frozen.vmd_xyz("icy.xyz");
    bouncer.vmd_xyz("tomato.xyz");
    remaining.name = "fluid";
    
    remaining.yg = 0.01;
    remaining.steps_per_dump = 5;
#ifdef USE_SEM
    runSemWithObstacles(L, Nm, n, dt, bouncer, remaining); 
#else
    remaining.equilibrate(0.1, 5000 * dt, dt, &bouncer);
#endif
}

    
