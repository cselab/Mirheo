#include <cstdlib>
#include <cmath>
#include <cstdio>
#include <cassert>

#include <algorithm>
#include <vector>
#include <random>

using namespace std;

inline float saru(unsigned int seed1, unsigned int seed2, unsigned int seed3)
{
    seed3 ^= (seed1<<7)^(seed2>>6);
    seed2 += (seed1>>4)^(seed3>>15);
    seed1 ^= (seed2<<9)+(seed3<<8);
    seed3 ^= 0xA5366B4D*((seed2>>11) ^ (seed1<<1));
    seed2 += 0x72BE1579*((seed1<<4)  ^ (seed3>>16));
    seed1 ^= 0X3F38A6ED*((seed3>>5)  ^ (((signed int)seed2)>>22));
    seed2 += seed1*seed3;
    seed1 += seed3 ^ (seed2>>2);
    seed2 ^= ((signed int)seed2)>>17;
    
    int state  = 0x79dedea3*(seed1^(((signed int)seed1)>>14));
    int wstate = (state + seed2) ^ (((signed int)state)>>8);
    state  = state + (wstate*(wstate^0xdddf97f5));
    wstate = 0xABCB96F7 + (wstate>>1);
    
    state  = 0x4beb5d59*state + 0x2600e1f7; // LCG
    wstate = wstate + 0x8009d14b + ((((signed int)wstate)>>31)&0xda879add); // OWS
    
    unsigned int v = (state ^ (state>>26))+wstate;
    unsigned int r = (v^(v>>20))*0x6957f5a7;
    
    float res = r / (4294967295.0f);
    return res;
}

struct Bouncer;

struct Particles
{
    static int idglobal;
    int n, saru_tag, myidstart;
    const float L;
    vector<float> xp, yp, zp, xv, yv, zv, xa, ya, za;

    vector<Bouncer *> bouncers;

    void aquire_global_id()
	{
	    myidstart = idglobal;
	    idglobal += n;
	}
    
    Particles (const int n, const float L);

    tuple<float, float, float, float> diag(FILE * f, float t);
    void _up(vector<float>& x, vector<float>& v, float f);
    void _up_enforce(vector<float>& x, vector<float>& v, float f);
    void _dpd_forces_bipartite(const float kBT, const double dt,
			       const float * const srcxp, const float * const srcyp, const float * const srczp,
			       const float * const srcxv, const float * const srcyv, const float * const srczv,
			       const int nsrc,
			       const int gididstart, const int gididend);
    void _dpd_forces(const float kBT, const double dt);
    void equilibrate(const float kBT, const double tend, const double dt);
    void vmd_xyz(const char * path, bool append = false);
    tuple<Particles, Particles> carve_zsandwich(const float half_width);

    void add_bouncer(Bouncer * b);
};

int Particles::idglobal;

struct Bouncer
{
    Particles frozen;

    Bouncer(Particles frozen): frozen(frozen) {}
    
    virtual void bounce(Particles& dest, const float dt) = 0;
};
    
Particles::Particles (const int n, const float L):
    n(n), L(L), xp(n), yp(n), zp(n), xv(n), yv(n), zv(n), xa(n), ya(n), za(n), saru_tag(0)
{
    if (n > 0)
	aquire_global_id();
    
    for(int i = 0; i < n; ++i)
    {
	xp[i] = -L * 0.5 + drand48() * L;
	yp[i] = -L * 0.5 + drand48() * L;
	zp[i] = -L * 0.5 + drand48() * L; 
    }
}

tuple<float, float, float, float> Particles::diag(FILE * f, float t)
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

    return make_tuple(T, xm, ym, zm);
}

void Particles::_up(vector<float>& x, vector<float>& v, float f)
{
    for(int i = 0; i < n; ++i)
	x[i] += f * v[i];
}

void Particles::_up_enforce(vector<float>& x, vector<float>& v, float f)
{
    for(int i = 0; i < n; ++i)
    {
	x[i] += f * v[i];
	x[i] -= L * floor(x[i] / L + 0.5);
    }
}

void Particles::_dpd_forces_bipartite(const float kBT, const double dt,
				      const float * const srcxp, const float * const srcyp, const float * const srczp,
				      const float * const srcxv, const float * const srcyv, const float * const srczv,
				      const int nsrc,
				      const int giddstart, const int gidsstart)
{
    const float xinvdomainsize = 1 / L;
    const float yinvdomainsize = 1 / L;
    const float zinvdomainsize = 1 / L;

    const float xdomainsize = L;
    const float ydomainsize = L;
    const float zdomainsize = L;

    const float invrc = 1.;
    const float gamma = 45;
    const float sigma = sqrt(2 * gamma * kBT);
    const float sigmaf = sigma / sqrt(dt);
    const float aij = 2.5;

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

void Particles::_dpd_forces(const float kBT, const double dt)
{
    for(int i = 0; i < n; ++i)
	xa[i] = ya[i] = za[i] = 0;
    
    _dpd_forces_bipartite(kBT, dt,
			  &xp.front(), &yp.front(), &zp.front(),
			  &xv.front(), &yv.front(), &zv.front(), n, myidstart, myidstart);

    for(int b = 0; b < bouncers.size(); ++b)
    {
	Particles src = bouncers[b]->frozen;
	
	_dpd_forces_bipartite(kBT, dt,
			      &src.xp.front(), &src.yp.front(), &src.zp.front(),
			      &src.xv.front(), &src.yv.front(), &src.zv.front(), src.n, myidstart, src.myidstart);
    }
}

void Particles::equilibrate(const float kBT, const double tend, const double dt)
{
    _dpd_forces(kBT, dt);

    vmd_xyz("ic.xyz", false);

    FILE * fdiag = fopen("diag-equilibrate.txt", "w");

    const size_t nt = (int)(tend / dt);

    for(int it = 0; it < nt; ++it)
    {
	if (it % 1 == 0)
	{
	    printf("step %d\n", it);
	    float t = it * dt;
	    diag(fdiag, t);
	    diag(stdout, t);
	}
		
	_up(xv, xa, dt * 0.5);
	_up(yv, ya, dt * 0.5);
	_up(zv, za, dt * 0.5);

	_up_enforce(xp, xv, dt);
	_up_enforce(yp, yv, dt);
	_up_enforce(zp, zv, dt);

	for(int b = 0; b < bouncers.size(); ++b)
	    bouncers[b]->bounce(*this, dt);
	
	_dpd_forces(kBT, dt);

	_up(xv, xa, dt * 0.5);
	_up(yv, ya, dt * 0.5);
	_up(zv, za, dt * 0.5);
	
	if (it % 30 == 0)
	    vmd_xyz("evolution.xyz", it > 0);
    }

    fclose(fdiag);
}
    
void Particles::vmd_xyz(const char * path, bool append)
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
    
tuple<Particles, Particles> Particles::carve_zsandwich(const float half_width)
{
    Particles partition[2] = {Particles(0, L), Particles(0, L)};
	    
    for(int i = 0; i < n; ++i)
    {
	const bool inside = fabs(zp[i]) <= half_width;

	partition[inside].xp.push_back(xp[i]);
	partition[inside].yp.push_back(yp[i]);
	partition[inside].zp.push_back(zp[i]);

	partition[inside].xv.push_back(xv[i]);
	partition[inside].yv.push_back(yv[i]);
	partition[inside].zv.push_back(zv[i]);

	partition[inside].xa.push_back(0);
	partition[inside].ya.push_back(0);
	partition[inside].za.push_back(0);
		
	partition[inside].n++;
    }

    partition[0].aquire_global_id();
    partition[1].aquire_global_id();
    
    return make_tuple(partition[0], partition[1]);
}

void Particles::add_bouncer(Bouncer* bouncer) { bouncers.push_back(bouncer); }

struct SandwichBouncer: Bouncer
{
    float half_width;
    
    SandwichBouncer(Particles& frozen, const float half_width):
	Bouncer(frozen), half_width(half_width)
	{
	    for(int i = 0; i < frozen.n; ++i)
		frozen.xv[i] = frozen.yv[i] = frozen.zv[i] = 0;
	}
    
    void bounce(Particles& dest, const float dt)
	{
	    for(int i = 0; i < dest.n; ++i)
	    {
		const float x = dest.xp[i];
		const float y = dest.yp[i];
		const float z = dest.zp[i];
		const float u = dest.xv[i];
		const float v = dest.yv[i];
		const float w = dest.zv[i];
		
		if (fabs(z) - half_width > 0)
		{
		    printf("particle %d shoud bounce back!\n", i);
		    const float xold = x - dt * u;
		    const float yold = y - dt * v;
		    const float zold = z - dt * w;

		    printf("old z %e old w %e -> colliding z %e\n", zold, w, dest.zp[i]);
		    printf("zold was inside? %e\n", fabs(zold) - half_width);
		    assert(fabs(zold) - half_width <= 0);
		    assert(fabs(w) > 0);

		    const float s = 1 - 2 * signbit(w);
		    const float t = (s * half_width - zold) / w;

		    printf("old z %f old w %f -> t %f\n", zold, w, t);
		    assert(t >= 0);
		    assert(t <= dt);
		    
		    const float lambda = 2 * t - dt;
		    
		    dest.xp[i] = xold + lambda * u;
		    dest.yp[i] = yold + lambda * v;
		    dest.zp[i] = zold + lambda * w;
		    printf("old z %e old w %e -> t %e -> new z %e\n", zold, w, t, dest.zp[i]);
		    assert(fabs(zold + lambda * w) - half_width <= 0);

		    dest.xv[i] = -u;
		    dest.yv[i] = -v;
		    dest.zv[i] = -w;

		    assert(!isnan(dest.xp[i]));
		    assert(!isnan(dest.yp[i]));
		    assert(!isnan(dest.zp[i]));
		    assert(!isnan(dest.xv[i]));
		    assert(!isnan(dest.yv[i]));
		    assert(!isnan(dest.zv[i]));
		}
	    }

	    printf("bounce done\n");
	}
};

int main()
{
    const float L = 10;
    const int Nm = 3;
    const int n = L * L * L * Nm;

    Particles particles(n, L);
    particles.equilibrate(.1, 5, 0.02);

    const float sandwich_half_width = L / 2 - 1.7;
    auto parts = particles.carve_zsandwich(sandwich_half_width);
    get<0>(parts).vmd_xyz("frozen.xyz");
    get<1>(parts).vmd_xyz("fluid.xyz");

    SandwichBouncer bouncer(get<0>(parts), sandwich_half_width);
    Particles remaining(get<1>(parts));
    remaining.add_bouncer(&bouncer);
    remaining.equilibrate(.1, 10, 0.02);
    printf("particles have been equilibrated");
}

    
