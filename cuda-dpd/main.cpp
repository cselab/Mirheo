#include <cstdlib>
#include <cmath>
#include <cstdio>
#include <cassert>

#include <algorithm>
#include <vector>
#include <random>

#include "dpd-cuda.h"

using namespace std;

typedef float real;

void vmd_xyz(const char * path, real * xyzuvw, const int n, bool append)
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
	fprintf(f, "1 %f %f %f\n",
		xyzuvw[0 + 6 * i],
		xyzuvw[1 + 6 * i],
		xyzuvw[2 + 6 * i]);
    
    fclose(f);

    printf("vmd_xyz: wrote to <%s>\n", path);
}

int main()
{
    real L = 10;

    const int Nm = 3;
    const int n = L * L * L * Nm;
    const real gamma = 45;
    const real sigma = 3;
    const real aij = 2.5;
    const real rc = 1;
    const bool cuda = true;
    const bool curand = true;
    const real dt = 0.02;
    const real tend = 10;
    
    vector<real> xyzuvw(6 * n), axayaz(3 * n);
    
    srand48(6516L);
    for(int i = 0; i < n; ++i)
    {
	const int cid = i / Nm;
	
	const int xcid = cid % (int)L;
	const int ycid = (cid / (int) L) % (int)L;
	const int zcid = (cid / (int) L / (int) L) % (int)L;

#if 1
	xyzuvw[0 + 6 * i] = -L * 0.5f +  drand48() * L;
	xyzuvw[1 + 6 * i] = -L * 0.5f +  drand48() * L;
	xyzuvw[2 + 6 * i] = -L * 0.5f +  drand48() * L;
#else
	xp[i] = -L * 0.5f + xcid + 0.5+ 0.15 * (1 - 2 * drand48() );
	yp[i] = -L * 0.5f + ycid + 0.5+ 0.15 * (1 - 2 * drand48() );
	zp[i] = -L * 0.5f + zcid + 0.5+ 0.15 * (1 - 2 * drand48() );
#endif
    }
    
    auto _diag = [&](FILE * f, float t)
	{
	    real sv2 = 0, xm = 0, ym = 0, zm = 0;
	    
	    for(int i = 0; i < n; ++i)
	    {
		sv2 += xyzuvw[3 + 6 * i] * xyzuvw[3 + 6 * i] + xyzuvw[4 + 6 * i] * xyzuvw[4 + 6 * i] + xyzuvw[5 + 6 * i] * xyzuvw[5 + 6 * i];
			
		xm += xyzuvw[3 + 6 * i];
		ym += xyzuvw[4 + 6 * i];
		zm += xyzuvw[5 + 6 * i];
	    }

	    real T = 0.5 * sv2 / (n * 3. / 2);

	    if (ftell(f) == 0)
		fprintf(f, "TIME\tkBT\tX-MOMENTUM\tY-MOMENTUM\tZ-MOMENTUM\n");

	    fprintf(f, "%s %+e\t%+e\t%+e\t%+e\t%+e\n", (f == stdout ? "DIAG:" : ""), t, T, xm, ym, zm);
	};
    
    auto _up_x = [&](real f)
	{
	    for(int i = 0; i < n; ++i)
	    {
		xyzuvw[0 + 6 * i] += f * xyzuvw[3 + 6 * i];
		xyzuvw[1 + 6 * i] += f * xyzuvw[4 + 6 * i];
		xyzuvw[2 + 6 * i] += f * xyzuvw[5 + 6 * i];
	    }

	    for(int i = 0; i < n; ++i)
	    {
		xyzuvw[0 + 6 * i] -= L * floor((xyzuvw[0 + 6 * i] + 0.5 * L) / L);
		xyzuvw[1 + 6 * i] -= L * floor((xyzuvw[1 + 6 * i] + 0.5 * L) / L);
		xyzuvw[2 + 6 * i] -= L * floor((xyzuvw[2 + 6 * i] + 0.5 * L) / L);

		assert(xyzuvw[0 + 6 * i] * 2  >= -L && xyzuvw[0 + 6 * i] * 2 < L); 
		assert(xyzuvw[1 + 6 * i] * 2  >= -L && xyzuvw[1 + 6 * i] * 2 < L); 
		assert(xyzuvw[2 + 6 * i] * 2  >= -L && xyzuvw[2 + 6 * i] * 2 < L); 
	    }
	};

    
    auto _up_v = [&](real f)
	{
	    for(int i = 0; i < n; ++i)
	    {
		xyzuvw[3 + 6 * i] += f * axayaz[0 + 3 * i];
		xyzuvw[4 + 6 * i] += f * axayaz[1 + 3 * i];
		xyzuvw[5 + 6 * i] += f * axayaz[2 + 3 * i];
	    }
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

    auto _f = [&]()
	{
	    if (cuda)
	    {
		vector<float> rsamples;

		if (!curand)
		{
		    rsamples.resize(n * 50);
		    
		    for(auto& e : rsamples)
			e = dgauss(gen);
		}
			
		forces_dpd_cuda(
		    &xyzuvw.front(),
		    &axayaz.front(),
		    NULL, n,
		    rc, L, L, L, aij, gamma, sigma, 1./sqrt(dt),
		    curand ? nullptr : &rsamples.front(), rsamples.size());
		
		return;
	    }

	    fill(axayaz.begin(), axayaz.end(), 0);
	    
	    for(int i = 0; i < n; ++i)
	    {
		for(int j = i + 1; j < n; ++j)
		{
		    real xr = xyzuvw[0 + 6 * i] - xyzuvw[0 + 6 * j];
		    real yr = xyzuvw[1 + 6 * i] - xyzuvw[1 + 6 * j];
		    real zr = xyzuvw[2 + 6 * i] - xyzuvw[2 + 6 * j];

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

		    real rdotv =
			xr * (xyzuvw[3 + 6 * i] - xyzuvw[3 + 6 * j]) +
			yr * (xyzuvw[4 + 6 * i] - xyzuvw[4 + 6 * j]) +
			zr * (xyzuvw[5 + 6 * i] - xyzuvw[5 + 6 * j]);
		    
		    real gij = dgauss(gen) / sqrt(dt);
		
		    real xf = (fc - gamma * wd * rdotv + sigma * wr * gij) * xr;
		    real yf = (fc - gamma * wd * rdotv + sigma * wr * gij) * yr;
		    real zf = (fc - gamma * wd * rdotv + sigma * wr * gij) * zr;

		    assert(!::isnan(xf));
		    assert(!::isnan(yf));
		    assert(!::isnan(zf));

		    axayaz[0 + 3 * i] += xf;
		    axayaz[1 + 3 * i] += yf;
		    axayaz[2 + 3 * i] += zf;

		    axayaz[0 + 3 * j] -= xf;
		    axayaz[1 + 3 * j] -= yf;
		    axayaz[2 + 3 * j] -= zf;
		}
	    }
	};
    
    _f();

    vmd_xyz("ic.xyz", &xyzuvw.front(), n, false);

    FILE * fdiag = fopen("diag.txt", "w");

    const size_t nt = (int)(tend / dt);

    for(int it = 0; it < nt; ++it)
    {
	if (it % 30 == 0)
	{
	    float t = it * dt;
	    _diag(fdiag, t);
	    _diag(stdout, t);
	}
		
	_up_v(dt * 0.5);

	_up_x(dt);
	
	_f();
	
	_up_v(dt * 0.5);
	
	if (it % 30 == 0)
	    vmd_xyz("evolution.xyz", &xyzuvw.front(), n, it > 0);
    }

    fclose(fdiag);
    
    return 0;
}
