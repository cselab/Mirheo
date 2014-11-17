#include <cstdlib>
#include <cmath>
#include <cstdio>
#include <cassert>

#include <algorithm>
#include <vector>
#include <random>

#include "cuda-dpd.h"

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
    real L = 40;

    const int Nm = 3;
    const int n = L * L * L * Nm;
    const real gamma = 45;
    const real sigma = 3;
    const real aij = 2.5;
    const real rc = 1;
    const bool cuda = true;
    const bool bipartite = false;
    const real dt = 0.02;
    const real tend = 0.4;//0.08 * 20;
    
    vector<real> xp(n), yp(n), zp(n), xv(n), yv(n), zv(n), xa(n), ya(n), za(n);
    srand48(6516L);
    for(int i = 0; i < n; ++i)
    {
	const int cid = i / Nm;
	
	const int xcid = cid % (int)L;
	const int ycid = (cid / (int) L) % (int)L;
	const int zcid = (cid / (int) L / (int) L) % (int)L;

#if 1
	xp[i] = -L * 0.5f +  drand48() * L;
	yp[i] = -L * 0.5f +  drand48() * L;
	zp[i] = -L * 0.5f +  drand48() * L;
	
#else
	xp[i] = -L * 0.5f + xcid + 0.5+ 0.015 * (1 - 2 * drand48() );
	yp[i] = -L * 0.5f + ycid + 0.5+ 0.015 * (1 - 2 * drand48() );
	zp[i] = -L * 0.5f + zcid + 0.5+ 0.015 * (1 - 2 * drand48() );
#endif
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
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> dgauss(0, 1);

    int cnt = 0;
    auto _f = [&]()
	{
	    fill(xa.begin(), xa.end(), 0);
	    fill(ya.begin(), ya.end(), 0);
	    fill(za.begin(), za.end(), 0);
	      
	    if (cuda) 
	    {
		if (!bipartite)
		    forces_dpd_cuda(
			&xp.front(), &yp.front(), &zp.front(),
			&xv.front(), &yv.front(), &zv.front(),
			&xa.front(), &ya.front(), &za.front(),
			n,
			rc, L, L, L, aij, gamma, sigma, 1./sqrt(dt));
		else
		{
		    const int pivot = n  * 0.5;//(int)(drand48() * xp.size());
		    		    
		    forces_dpd_cuda_bipartite(
			&xp.front(), &yp.front(), &zp.front(),
			&xv.front(), &yv.front(), &zv.front(),
			&xa.front(), &ya.front(), &za.front(),
			pivot, 0,
			&xp.front() + pivot, &yp.front() + pivot, &zp.front() + pivot,
			&xv.front() + pivot, &yv.front() + pivot, &zv.front() + pivot,
			&xa.front() + pivot, &ya.front() + pivot, &za.front() + pivot,
			n - pivot, pivot,
			rc, L, L, L, aij, gamma, sigma, 1./sqrt(dt));

		      forces_dpd_cuda(
			&xp.front(), &yp.front(), &zp.front(),
			&xv.front(), &yv.front(), &zv.front(),
			&xa.front(), &ya.front(), &za.front(),
			pivot,
			rc, L, L, L, aij, gamma, sigma, 1./sqrt(dt));

		      forces_dpd_cuda(
			  &xp.front() + pivot, &yp.front() + pivot, &zp.front() + pivot,
			  &xv.front() + pivot, &yv.front() + pivot, &zv.front() + pivot,
			  &xa.front() + pivot, &ya.front() + pivot, &za.front() + pivot,
			  n - pivot,
			  rc, L, L, L, aij, gamma, sigma, 1./sqrt(dt));
		}
		
		return;
	    }

	    fill(xa.begin(), xa.end(), 0);
	    fill(ya.begin(), ya.end(), 0);
	    fill(za.begin(), za.end(), 0);
	    
	    for(int i = 0; i < n; ++i)
	    {
		for(int j = i + 1; j < n; ++j)
		{
		    real xr = xp[i] - xp[j];
		    real yr = yp[i] - yp[j];
		    real zr = zp[i] - zp[j];

		    xr -= L * floor(0.5 + xr / L);
		    yr -= L * floor(0.5 + yr / L);
		    zr -= L * floor(0.5 + zr / L);

		    real rij = sqrt(xr * xr + yr * yr + zr * zr);

		    xr /= rij;
		    yr /= rij;
		    zr /= rij;
		
		    real rc = 1; 
		    real fc = max((real)0, aij * (1 - rij / rc));

		    real wr = max((real)0, 1 - rij / rc);
		    real wd = wr * wr;

		    real rdotv = xr * (xv[i] - xv[j]) + yr * (yv[i] - yv[j]) + zr * (zv[i] - zv[j]);
		    real gij = dgauss(gen) / sqrt(dt);
		
		    real xf = (fc - gamma * wd * rdotv + sigma * wr * gij) * xr;
		    real yf = (fc - gamma * wd * rdotv + sigma * wr * gij) * yr;
		    real zf = (fc - gamma * wd * rdotv + sigma * wr * gij) * zr;

		    assert(!::isnan(xf));
		    assert(!::isnan(yf));
		    assert(!::isnan(zf));

		    xa[i] += xf;
		    ya[i] += yf;
		    za[i] += zf;

		    xa[j] -= xf;
		    ya[j] -= yf;
		    za[j] -= zf;
		}
	    }
	};
    
    _f();

    

    vmd_xyz("ic.xyz", &xp.front(), &yp.front(), &zp.front(), n, false);

    FILE * fdiag = fopen("diag.txt", "w");

    const size_t nt = (int)(tend / dt);

    for(int it = 0; it < nt; ++it)
    {
	//const double dt = 0;
	if (it % 30 == 0)
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

	_up(xv, xa, dt * 0.5);
	_up(yv, ya, dt * 0.5);
	_up(zv, za, dt * 0.5);
	
	if (it % 30 == 0)
	    vmd_xyz("evolution.xyz", &xp.front(), &yp.front(), &zp.front(), n, it > 0);
    }

    fclose(fdiag);

    printf("done with %zd iterations.\n", nt);	    
    return 0;
}
