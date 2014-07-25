#include <cstdlib>
#include <cmath>
#include <cstdio>
#include <cassert>

#include <algorithm>
#include <vector>
#include <tuple>

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
    int XL = 10;
    int YL = 10;
    int ZL = 10;
    const int rho = 3;
    const int n = XL * YL * ZL * rho;
    const real gamma = 4.5;
    const real sigma = 3;
    const real aij = 25;
    const real rc = 1;
    const real dt = 0.02;
    const real tend = 100;
    const int steps_per_dump = 50;
    vector<real> xp(n), yp(n), zp(n), xv(n), yv(n), zv(n), xa(n), ya(n), za(n);
    
    for(int i = 0; i < n; ++i)
    {
	xp[i] = -XL * 0.5 + drand48() * XL;
	yp[i] = -YL * 0.5 + drand48() * YL;
	zp[i] = -ZL * 0.5 + drand48() * ZL;
    }

    vector< tuple<real, int> > xvprofile(100);
    		
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

	    if (t > tend * 0.5)
	    {
		const int nbins = xvprofile.size();
		const real h = XL / (double)xvprofile.size();
		
		for(int i = 0; i < n; ++i)
		{
		    const int idbin = (nbins + (int)((xp[i] + 0.5 * XL) / h)) % nbins;
		    
		    auto bin = xvprofile[idbin];

		    xvprofile[idbin] = make_tuple(get<0>(bin) + yv[i], get<1>(bin) + 1);
		}

		FILE * f = fopen("yv-profile.dat", "w");
		int c = 0;
		for(auto e :xvprofile)
		    fprintf(f, "%e %e\n", h * (0.5 + c++) - 0.5 * XL, get<0>(e) / max(1, get<1>(e)));
		
		fclose(f);
	    }
	};
    
    auto _up = [=](vector<real>& x, vector<real>& v, real f)
	{
	    for(int i = 0; i < n; ++i)
		x[i] += f * v[i];
	};

    auto _up_enforce = [=](vector<real>& x, vector<real>& v, real f, real L)
	{
	    for(int i = 0; i < n; ++i)
	    {
		x[i] += f * v[i];
		x[i] -= L * floor((x[i] + 0.5 * L) / L);
		
	    }
	};

    auto _f = [&]()
	{
	      forces_dpd_cuda(
		    &xp.front(), &yp.front(), &zp.front(),
		    &xv.front(), &yv.front(), &zv.front(),
		    &xa.front(), &ya.front(), &za.front(),
		    nullptr, n,
		    rc, XL, YL, ZL, aij, gamma, sigma, 1./sqrt(dt));

	    for(int i = 0; i < n; ++i)
		ya[i] += 0.1 * (2 * (xp[i] > 0) - 1);
	};
    
    _f();

    vmd_xyz("ic.xyz", &xp.front(), &yp.front(), &zp.front(), n, false);

    FILE * fdiag = fopen("diag.txt", "w");

    const size_t nt = (int)(tend / dt);

    for(int it = 0; it < nt; ++it)
    {
	if (it % 50 == 0)
	{
	    float t = it * dt;
	    _diag(fdiag, t);
	    _diag(stdout, t);
	}
		
	_up(xv, xa, dt * 0.5);
	_up(yv, ya, dt * 0.5);
	_up(zv, za, dt * 0.5);

	_up_enforce(xp, xv, dt, XL);
	_up_enforce(yp, yv, dt, YL);
	_up_enforce(zp, zv, dt, ZL);
	
	_f();

	_up(xv, xa, dt * 0.5);
	_up(yv, ya, dt * 0.5);
	_up(zv, za, dt * 0.5);
	
	if (it % steps_per_dump == 0)
	    vmd_xyz("evolution.xyz", &xp.front(), &yp.front(), &zp.front(), n, it > 0);
    }

    fclose(fdiag);
    
    return 0;
}
