#include <cstdlib>
#include <cmath>
#include <cstdio>
#include <cassert>

#include <algorithm>
#include <vector>
#include <random>

using namespace std;

typedef double real;

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
    const int nx = 8;
    const int n = nx * nx * nx;
    
    vector<real> xp(n), yp(n), zp(n), xv(n), yv(n), zv(n), xa(n), ya(n), za(n);

    real domain_size = 2;
    real h = domain_size / nx;
    for(int i = 0; i < n; ++i)
    {
	xp[i] = -domain_size * 0.5f + (i % nx) * h;
	yp[i] = -domain_size * 0.5f + (i / nx % nx) * h;
	zp[i] = -domain_size * 0.5f + (i / nx / nx % nx) * h;
    }
   
    const real dt = 1e-4; //?
    const real tend = 1; //?
    const size_t nt = (int)(tend / dt);

    auto _diag = [&](FILE * f)
	{
	    real sv2 = 0, xm = 0, ym = 0, zm = 0;
	    
	    for(int i = 0; i < n; ++i)
	    {
		sv2 += xv[i] * xv[i] + yv[i] * yv[i] + zv[i] * zv[i];
		
		xm += xv[i];
		ym += yv[i];
		zm += zv[i];
	    }

	    real kB = 1;
	    real T = 0.5 * sv2 / (n * kB * 3. / 2);

	    if (ftell(f) == 0)
		fprintf(f, "TEMPERATURE\tX-MOMENTUM\tY-MOMENTUM\tZ-MOMENTUM\n");

	    fprintf(f, "%s %+e\t%+e\t%+e\t%+e\n", (f == stdout ? "DIAG:" : ""), T, xm, ym, zm);
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
		x[i] -= domain_size * floor(x[i] / domain_size + 0.5);
	    }
	};
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> dgauss(0, 1);

    auto _f = [&]()
	{
	    real gamma = 1; //?
	    real sigma = sqrt(2 * gamma * /*kB * T*/ 1); //?
	
	    //real kappa = ?;
	    //real Nm = ?;
	    //real alpha = ?;
	    //real rho = ?;
	    real aij = 0.1;// (1./ kappa * Nm - 1) / (2 * alpha * rho);

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

		    xr -= domain_size * floor(0.5f + xr / domain_size);
		    yr -= domain_size * floor(0.5f + yr / domain_size);
		    zr -= domain_size * floor(0.5f + zr / domain_size);

		    real rij = sqrt(xr * xr + yr * yr + zr * zr);
		
		    real rc = 10; //?
		    real fc = max((real)0, aij * (1 - rij / rc));

		    real wr = max((real)0, 1 - rij / rc);
		    real wd = wr * wr;

		    real rdotv = xr * (xv[i] - xv[j]) + yr * (yv[i] - yv[j]) + zr * (zv[i] - zv[j]);
		    real gij = dgauss(gen) / sqrt(dt);
		
		    real xf = (fc - gamma * wd * rdotv + sigma * wr * gij) * xr / rij;
		    real yf = (fc - gamma * wd * rdotv + sigma * wr * gij) * yr / rij;
		    real zf = (fc - gamma * wd * rdotv + sigma * wr * gij) * zr / rij;

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
    
    for(int it = 0; it < 6000; ++it)
    {
	if (it % 10 == 0)
	{
	    _diag(fdiag);
	    _diag(stdout);
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
    
    return 0;
}
