/*
 *  main.cpp
 *  Part of CTC/cuda-dpd-sem/sem/
 *
 *  Created and authored by Diego Rossinelli on 2014-07-10.
 *  Copyright 2015. All rights reserved.
 *
 *  Users are NOT authorized
 *  to employ the present software for their own publications
 *  before getting a written permission from the author of this file.
 */

#include <cstdlib>
#include <cmath>
#include <cstdio>
#include <cassert>

#include <algorithm>
#include <vector>
#include <random>

#include "cuda-sem.h"

using namespace std;

typedef float real;

void vmd_xyz(const char * path, real * xs, real * ys, real * zs, const int n, bool append)
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
	fprintf(f, "1 %f %f %f\n", xs[i], ys[i], zs[i]);
    
    fclose(f);

    printf("vmd_xyz: wrote to <%s>\n", path);
}

int main()
{
    //np,  rc,  LX, LY, LZ,  gamma, temp, dt,   u0,    rho,  req, D
    //1e3, 1.0, 10, 10, 10,  80,    0.1,  0.01, 0.001, 1.5,  0.85, 0.0001
    
    real L = 7;

    const float Nm = 5.8;
    const int n = L * L * L * Nm;
    const real rcutoff = 2.5;
    const real gamma = 80, temp = 0.1, dt = 0.01, u0 = 0.001, rho = 1.5, req = 0.85, D = .0001, rc = 1;
    const bool cuda = true;
    const real tend = 10;//200;//0.08 * 20;
    
    vector<real> xp(n), yp(n), zp(n), xv(n), yv(n), zv(n), xa(n), ya(n), za(n), xb(n), yb(n), zb(n);
    srand48(6516L);
    for(int i = 0; i < n; ++i)
    {
	xp[i] = -L * 0.5f +  drand48() * L;
	yp[i] = -L * 0.5f +  drand48() * L;
	zp[i] = -L * 0.5f +  drand48() * L;	
    }
    
    auto _diag = [&](FILE * f, float t)
	{
	    real sv2 = 0, xm = 0, ym = 0, zm = 0;
	    
	    for(int i = 0; i < n; ++i)
	    {
		sv2 += xv[i] * xv[i] + yv[i] * yv[i] + zv[i] * zv[i];
		
		xm += xv[i];
		ym += yv[i];
		zm += zv[i];
	    }

	    real T = 0.5 * sv2 / (n * 3. / 2);

	    if (ftell(f) == 0)
		fprintf(f, "TIME\tkBT\tX-MOMENTUM\tY-MOMENTUM\tZ-MOMENTUM\n");

	    fprintf(f, "%s %+e\t%+e\t%+e\t%+e\t%+e\n", (f == stdout ? "DIAG:" : ""), t, T, xm, ym, zm);
	};
    
    auto _up = [=](vector<real>& x, vector<real>& v, real f)
	{
	    for(int i = 0; i < n; ++i)
		x[i] += f * v[i];
	};

    auto _up_enforce = [=](vector<real>& x, vector<real>& v, real f)
	{
	    for(int i = 0; i < n; ++i)
	    {
		x[i] += f * v[i];
		x[i] -= L * floor((x[i] + 0.5 * L) / L);
		
	    }
	};

    int cnt = 0; 
    auto _f = [&]()
	{
	    fill(xa.begin(), xa.end(), 0);
	    fill(ya.begin(), ya.end(), 0);
	    fill(za.begin(), za.end(), 0);

	    fill(xb.begin(), xb.end(), 0);
		fill(yb.begin(), yb.end(), 0);
		fill(zb.begin(), zb.end(), 0);
	      	  
	    forces_sem_cuda(
		&xp.front(), &yp.front(), &zp.front(),
		&xv.front(), &yv.front(), &zv.front(),
		&xa.front(), &ya.front(), &za.front(),
		n,
		rcutoff, L, L, L, gamma, temp, dt, u0, rho, req, D, rc);

	    forces_sem_cuda_direct(
		&xp.front(), &yp.front(), &zp.front(),
		&xv.front(), &yv.front(), &zv.front(),
		&xb.front(), &yb.front(), &zb.front(),
		n,
		rcutoff, L, L, L, gamma, temp, dt, u0, rho, req, D, rc);

	    for (int i=0; i<n; i++)
	    {
	    	if (true) printf("%4d:  X:  expected  %.7f,  got %.7f\n", i, xa[i], xb[i]);
	    	if (fabs(ya[i]-yb[i]) > 1e-7) printf("%4d:  Y:  expected  %.7f,  got %.7f\n", i, ya[i], yb[i]);
	    	if (fabs(za[i]-zb[i]) > 1e-7) printf("%4d:  Z:  expected  %.7f,  got %.7f\n", i, za[i], zb[i]);
	    }
	};
    
    _f();

    vmd_xyz("ic.xyz", &xp.front(), &yp.front(), &zp.front(), n, false);

    FILE * fdiag = fopen("diag.txt", "w");

    const size_t nt = (int)(tend / dt);

    for(int it = 0; it < nt; ++it)
    {
	//const double dt = 0;
	if (it % 100 == 0)
	{
	    float t = it * dt;
	    _diag(fdiag, t);
	    _diag(stdout, t);
	}
		
	_up(xv, xa, dt * 0.5);
	_up(yv, ya, dt * 0.5);
	_up(zv, za, dt * 0.5);

	_up_enforce(xp, xv, dt);
	_up_enforce(yp, yv, dt);
	_up_enforce(zp, zv, dt);
	
	_f();
	exit(0);

	_up(xv, xa, dt * 0.5);
	_up(yv, ya, dt * 0.5);
	_up(zv, za, dt * 0.5);
	
	if (it % 100 == 0)
	    vmd_xyz("evolution.xyz", &xp.front(), &yp.front(), &zp.front(), n, it > 0);
    }

    fclose(fdiag);
    
    return 0;
}
