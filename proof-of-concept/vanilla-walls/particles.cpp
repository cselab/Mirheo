/*
 *  particles.cpp
 *  Part of uDeviceX/vanilla-walls/
 *
 *  Created by Kirill Lykov, on 2014-08-12.
 *  Authored by Diego Rossinelli on 2014-08-12.
 *  Copyright 2015. All rights reserved.
 *
 *  Users are NOT authorized
 *  to employ the present software for their own publications
 *  before getting a written permission from the author of this file.
 */
#include "particles.h"

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
#include "funnel-bouncer.h"

using namespace std;

#include <sys/time.h>

class Timer
{
    struct timeval t_start, t_end;
    struct timezone t_zone;
	
public:
	
    void start()
	{
	    gettimeofday(&t_start,  &t_zone);
	}
	
    double stop()
	{
	    gettimeofday(&t_end,  &t_zone);
	    return (t_end.tv_usec  - t_start.tv_usec)*1e-6  + (t_end.tv_sec  - t_start.tv_sec);
	}
};

const std::string Particles::outFormat = "dump";

void Particles::internal_forces_bipartite(const float kBT, const double dt,
               const float * const srcxp, const float * const srcyp, const float * const srczp,
               const float * const srcxv, const float * const srcyv, const float * const srczv,
               const int nsrc,
               const int giddstart, const int gidsstart)
{
    const float xdomainsize = L[0];
    const float ydomainsize = L[1];
    const float zdomainsize = L[2];

    const float xinvdomainsize = 1 / xdomainsize;
    const float yinvdomainsize = 1 / ydomainsize;
    const float zinvdomainsize = 1 / zdomainsize;

    const float invrc = 1.;
    const float gamma = 45;
    const float sigma = sqrt(2 * gamma * kBT);
    const float sigmaf = sigma / sqrt(dt);
    const float aij = 2.5;

    if (cuda)
    {
	if(srcxp == &xp.front())
	    forces_dpd_cuda(&xp.front(), &yp.front(), &zp.front(),
			    &xv.front(), &yv.front(), &zv.front(),
			    &xa.front(), &ya.front(), &za.front(),
			    n,
			    1, xdomainsize, ydomainsize, zdomainsize,
			    aij,  gamma,  sigma,  1 / sqrt(dt));
	else
	    forces_dpd_cuda_bipartite(&xp.front(), &yp.front(), &zp.front(),
				      &xv.front(), &yv.front(), &zv.front(),
				      &xa.front(), &ya.front(), &za.front(),
				      n, giddstart,
				      srcxp,  srcyp,  srczp,
				      srcxv,  srcyv,  srczv,
				      NULL, NULL, NULL,
				      nsrc, gidsstart,
				      1, xdomainsize, ydomainsize, zdomainsize,
				      aij,  gamma,  sigma,  1 / sqrt(dt));
    }
    else
    {

#pragma omp parallel for
	for(int i = 0; i < n; ++i)
	{
	    float xf = 0, yf = 0, zf = 0;

	    const int dpid = giddstart + i;

	    for(int j = 0; j < nsrc; ++j)
	    {
		const int spid = gidsstart + j;

		if (spid == dpid)
		    continue;

		const float xdiff = xp[i] - srcxp[j];
		const float ydiff = yp[i] - srcyp[j];
		const float zdiff = zp[i] - srczp[j];

		const float _xr = xdiff - xdomainsize * floorf(0.5f + xdiff * xinvdomainsize);
		const float _yr = ydiff - ydomainsize * floorf(0.5f + ydiff * yinvdomainsize);
		const float _zr = zdiff - zdomainsize * floorf(0.5f + zdiff * zinvdomainsize);

		const float rij2 = _xr * _xr + _yr * _yr + _zr * _zr;
		float invrij = 1./sqrtf(rij2);

		if (rij2 == 0)
		    invrij = 100000;

		const float rij = rij2 * invrij;
		const float wr = max((float)0, 1 - rij * invrc);

		const float xr = _xr * invrij;
		const float yr = _yr * invrij;
		const float zr = _zr * invrij;

		const float rdotv =
		    xr * (xv[i] - srcxv[j]) +
		    yr * (yv[i] - srcyv[j]) +
		    zr * (zv[i] - srczv[j]);

		const float mysaru = saru(min(spid, dpid), max(spid, dpid), saru_tag);
		const float myrandnr = 3.464101615f * mysaru - 1.732050807f;

		const float strength = (aij - gamma * wr * rdotv + sigmaf * myrandnr) * wr;

		xf += strength * xr;
		yf += strength * yr;
		zf += strength * zr;
	    }

	    xa[i] += xf;
	    ya[i] += yf;
	    za[i] += zf;
	}

	saru_tag++;
    }
}

void Particles::acquire_global_id()
{
    myidstart = idglobal;
    idglobal += n;
}

Particles::Particles (const int n, const float Lx)
: Particles(n, Lx, Lx, Lx)
{}

Particles::Particles (const int n, const float Lx, const float Ly, const float Lz)
    : n(n), L{Lx, Ly, Lz}, saru_tag(0), xp(n), yp(n), zp(n), xv(n), yv(n), zv(n), xa(n), ya(n), za(n)
{
    if (n > 0)
    acquire_global_id();

    for(int i = 0; i < n; ++i)
    {
    xp[i] = -L[0] * 0.5 + drand48() * L[0];
    yp[i] = -L[1] * 0.5 + drand48() * L[1];
    zp[i] = -L[2] * 0.5 + drand48() * L[2];
    }
}

void Particles::diag(FILE * f, float t)
{
    float sv2 = 0, xm = 0, ym = 0, zm = 0;

    for(int i = 0; i < n; ++i)
    {
    sv2 += xv[i] * xv[i] + yv[i] * yv[i] + zv[i] * zv[i];

    xm += xv[i];
    ym += yv[i];
    zm += zv[i];
    }

    float T = 0.5 * sv2 / (n * 3. / 2);

    if (ftell(f) == 0)
    fprintf(f, "TIME\tkBT\tX-MOMENTUM\tY-MOMENTUM\tZ-MOMENTUM\n");

    fprintf(f, "%s %+e\t%+e\t%+e\t%+e\t%+e\n", (f == stdout ? "DIAG:" : ""), t, T, xm, ym, zm);
}

void Particles::vmd_xyz(const char * path, bool append) const
{
    FILE * f = fopen(path, append ? "a" : "w");

    if (f == NULL)
    {
    printf("I could not open the file <%s>\n", path);
    printf("Aborting now.\n");
    abort();
    }

    fprintf(f, "%d\n", n);
    fprintf(f, "mymolecule\n");

    for(int i = 0; i < n; ++i)
    fprintf(f, "1 %f %f %f\n", xp[i], yp[i], zp[i]);

    fclose(f);

    printf("vmd_xyz: wrote to <%s>\n", path);
}

// might be opened by OVITO and xmovie (xwindow-based utility)
void Particles::lammps_dump(const char* path, size_t timestep) const
{
  bool append = timestep > 0;
  FILE * f = fopen(path, append ? "a" : "w");

  if (f == NULL)
  {
  printf("I could not open the file <%s>\n", path);
  printf("Aborting now.\n");
  abort();
  }

  // header
  fprintf(f, "ITEM: TIMESTEP\n%lu\n", timestep);
  fprintf(f, "ITEM: NUMBER OF ATOMS\n%d\n", n);
  fprintf(f, "ITEM: BOX BOUNDS pp pp pp\n%g %g\n%g %g\n%g %g\n",
      -L[0]/2.0, L[0]/2.0, -L[1]/2.0, L[1]/2.0, -L[2]/2.0, L[2]/2.0);

  fprintf(f, "ITEM: ATOMS id type xs ys zs\n");

  // positions <ID> <type> <x> <y> <z>
  // free particles have type 2, while rings 1
  size_t type = 1; //skip for now
  for (int i = 0; i < n; ++i) {
    fprintf(f, "%lu %lu %g %g %g\n", i, type, xp[i], yp[i], zp[i]);
  }

  fclose(f);
}

int Particles::idglobal;

void Particles::internal_forces(const float kBT, const double dt)
{
    for(int i = 0; i < n; ++i)
    {
    xa[i] = xg;
    ya[i] = yg;
    za[i] = zg;
    }

    internal_forces_bipartite(kBT, dt,
              &xp.front(), &yp.front(), &zp.front(),
              &xv.front(), &yv.front(), &zv.front(), n, myidstart, myidstart);
}

void Particles::equilibrate(const float kBT, const double tend, const double dt, const Bouncer* bouncer)
{
    auto _up  = [&](vector<float>& x, vector<float>& v, float f)
    {
        for(int i = 0; i < n; ++i)
        x[i] += f * v[i];
    };

    auto _up_enforce = [&](vector<float>& x, vector<float>& v, float f, float boxLength)
    {
        for(int i = 0; i < n; ++i)
        {
        x[i] += f * v[i];
        x[i] -= boxLength * floor(x[i] / boxLength + 0.5);
        }
    };

    internal_forces(kBT, dt);

    // don't want to have any output in this case
    if (steps_per_dump != std::numeric_limits<int>::max())
        vmd_xyz("ic.xyz", false);

    FILE * fdiag = fopen("diag-equilibrate.txt", "w");

    const size_t nt = (int)(tend / dt);

    for(int it = 0; it < nt; ++it)
    {
    if (it % steps_per_dump == 0 && it != 0)
    {
        printf("step %d\n", it);
        float t = it * dt;
        diag(fdiag, t);
        diag(stdout, t);
    }

    _up(xv, xa, dt * 0.5);
    _up(yv, ya, dt * 0.5);
    _up(zv, za, dt * 0.5);

    _up_enforce(xp, xv, dt, L[0]);
    _up_enforce(yp, yv, dt, L[1]);
    _up_enforce(zp, zv, dt, L[2]);

    if (bouncer != nullptr)
        bouncer->bounce(*this, dt);

    internal_forces(kBT, dt);

    if (bouncer != nullptr)
        bouncer->compute_forces(kBT, dt, *this);

    _up(xv, xa, dt * 0.5);
    _up(yv, ya, dt * 0.5);
    _up(zv, za, dt * 0.5);

    if (it % steps_per_dump == 0)
        lammps_dump("evolution.dump", it);
        vmd_xyz((name == "" ? "evolution.xyz" : (name + "-evolution.xyz")).c_str(), it > 0);
    }

    fclose(fdiag);
}


