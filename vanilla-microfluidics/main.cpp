/*
 *  main.cpp
 *  Part of CTC/vanilla-microfluidics/
 *
 *  Created and authored by Diego Rossinelli on 2014-08-08.
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

#include <vector>
#include <string>

#include "cuda-dpd.h"
#include "funnel-obstacle.h"
#include "cell-factory.h"
#include "cuda-sem.h"



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

using namespace std;

struct Bouncer;

struct Particles
{
    static int idglobal;
    mutable int saru_tag;
    int n,  myidstart, steps_per_dump = 100;
    float L, xg = 0, yg = 0, zg = 0;
    vector<float> xp, yp, zp, xv, yv, zv, xa, ya, za;
    Bouncer * bouncer = nullptr;
    string name;

 void _dpd_forces_1particle(const float kBT, const double dt,
            int i, const float* offset,
            const float* coord, const float* vel, // float3 arrays
            float* df, // float3 for delta{force}
            const int giddstart) const
    {
        const float xinvdomainsize = 1.0f / L;
        const float yinvdomainsize = 1.0f / L;
        const float zinvdomainsize = 1.0f / L;

        const float xdomainsize = L;
        const float ydomainsize = L;
        const float zdomainsize = L;

        const float invrc = 1.0f;
        const float gamma = 45.0f;
        const float sigma = sqrt(2.0f * gamma * kBT);
        const float sigmaf = sigma / sqrt(dt);
        const float aij = 2.5f;

        float xf = 0, yf = 0, zf = 0;

        const int dpid = giddstart + i; //doesn't matter since compute only i->j

        for(int j = 0; j < n; ++j)
        {
            //if (i == 3)
            //    std::cout << j << std::endl;

            const int spid = myidstart + j;

            if (spid == dpid)
            continue;

            const float xdiff = coord[0] - (xp[j] + offset[0]);
            const float ydiff = coord[1] - (yp[j] + offset[1]);
            const float zdiff = coord[2] - (zp[j] + offset[2]);

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

            assert(xv[j] == 0 && yv[j] == 0 && zv[j] == 0);
            const float rdotv =
            xr * (vel[0] - xv[j]) +
            yr * (vel[1] - yv[j]) +
            zr * (vel[2] - zv[j]);

            const float mysaru = saru(min(spid, dpid), max(spid, dpid), saru_tag);
            const float myrandnr = 3.464101615f * mysaru - 1.732050807f;

            const float strength = (aij - gamma * wr * rdotv + sigmaf * myrandnr) * wr;

            xf += strength * xr;
            yf += strength * yr;
            zf += strength * zr;
        }

        df[0] += xf;
        df[1] += yf;
        df[2] += zf;
    }
    
    void internal_forces_bipartite(const float kBT, const double dt,
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



#if 1
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
#else
	  
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
#endif
	}
    
    void acquire_global_id()
	{
	    myidstart = idglobal;
	    idglobal += n;
	}
    
    Particles (const int n, const float L):
	n(n), L(L), xp(n), yp(n), zp(n), xv(n), yv(n), zv(n), xa(n), ya(n), za(n), saru_tag(0)
	{
	    if (n > 0)
		acquire_global_id();
    
	    for(int i = 0; i < n; ++i)
	    {
		xp[i] = -L * 0.5 + drand48() * L;
		yp[i] = -L * 0.5 + drand48() * L;
		zp[i] = -L * 0.5 + drand48() * L;

		xv[i] = yv[i] = zv[i] = 0;
	    }
	}
    
    void diag(FILE * f, float t)
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
       
    void vmd_xyz(const char * path, bool append = false)
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

   void _up (vector<float>& x, vector<float>& v, float f)
	{
	    for(int i = 0; i < n; ++i)
		x[i] += f * v[i];
	}

    void _up_enforce(vector<float>& x, vector<float>& v, float f)
	{
	    for(int i = 0; i < n; ++i)
	    {
		x[i] += f * v[i];
		x[i] -= L * floor(x[i] / L + 0.5);
	    }
	}

    void update_stage1(const float dt, const int timestepid);

    void update_stage2 (const float dt, const int timestepid)
	{
	    _up(xv, xa, dt * 0.5);
	    _up(yv, ya, dt * 0.5);
	    _up(zv, za, dt * 0.5);

	    if (timestepid % steps_per_dump == 0)
		vmd_xyz((name == "" ? "evolution.xyz" : (name + "-evolution.xyz")).c_str(), timestepid > 0);
	};
    
    virtual void internal_forces(const float kBT, const double dt);
    void equilibrate(const float kBT, const double tend, const double dt);
};

int Particles::idglobal;

#if 1
struct Bouncer
{
    Particles frozen;
    Bouncer(const float L): frozen(0, L) {}
    virtual ~Bouncer() {}
    virtual void _mark(bool * const freeze, Particles p) = 0;
    virtual void bounce(Particles& dest, const float dt) = 0;
    virtual void compute_forces(const float kBT, const double dt, Particles& freeParticles) ;
    
    virtual Particles carve(const Particles& p) //TODO consider moving to Sandwich the implementation
	{
	    bool * const freeze = new bool[p.n];
	    
	    _mark(freeze, p);
	    
	    Particles partition[2] = {Particles(0, p.L), Particles(0, p.L)};
	    
	    splitParticles(p, freeze, partition);

	    frozen = partition[0];
	    frozen.name = "frozen";
	    
	    for(int i = 0; i < frozen.n; ++i)
		frozen.xv[i] = frozen.yv[i] = frozen.zv[i] = 0;

	    delete [] freeze;
	    
	    return partition[1];
	}

    template<typename MaskType>
    static void splitParticles(const Particles& p, const MaskType& freezeMask,
            Particles* partition) // Particles[2] array
    {
        for(int i = 0; i < p.n; ++i)
        {
            const int slot = !freezeMask[i];

            partition[slot].xp.push_back(p.xp[i]);
            partition[slot].yp.push_back(p.yp[i]);
            partition[slot].zp.push_back(p.zp[i]);

            partition[slot].xv.push_back(p.xv[i]);
            partition[slot].yv.push_back(p.yv[i]);
            partition[slot].zv.push_back(p.zv[i]);

            partition[slot].xa.push_back(0);
            partition[slot].ya.push_back(0);
            partition[slot].za.push_back(0);

            partition[slot].n++;
        }

        partition[0].acquire_global_id();
        partition[1].acquire_global_id();
    }
};
#else
struct Bouncer
{
    Particles frozen;
    Bouncer(const float L): frozen(0, L) {}

    virtual void _mark(bool * const freeze, Particles p) = 0;
    virtual void bounce(Particles& dest, const float dt) = 0;
    
    Particles carve(const Particles p)
	{
	    bool * const freeze = new bool[p.n];
	    
	    _mark(freeze, p);
	    
	    Particles partition[2] = {Particles(0, p.L), Particles(0, p.L)};
	    
	    for(int i = 0; i < p.n; ++i)
	    {
		const int slot = !freeze[i];
		
		partition[slot].xp.push_back(p.xp[i]);
		partition[slot].yp.push_back(p.yp[i]);
		partition[slot].zp.push_back(p.zp[i]);
			  	       
		partition[slot].xv.push_back(p.xv[i]);
		partition[slot].yv.push_back(p.yv[i]);
		partition[slot].zv.push_back(p.zv[i]);
			  
		partition[slot].xa.push_back(0);
		partition[slot].ya.push_back(0);
		partition[slot].za.push_back(0);
			  
		partition[slot].n++;
	    }

	    partition[0].acquire_global_id();
	    partition[1].acquire_global_id();

	    frozen = partition[0];
	    frozen.name = "frozen";
	    
	    for(int i = 0; i < frozen.n; ++i)
		frozen.xv[i] = frozen.yv[i] = frozen.zv[i] = 0;

	    delete [] freeze;
	    
	    return partition[1];
	}
};
#endif

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

    if (bouncer != nullptr)
    {
	Particles& src = bouncer->frozen;
	
	internal_forces_bipartite(kBT, dt,
			      &src.xp.front(), &src.yp.front(), &src.zp.front(),
			      &src.xv.front(), &src.yv.front(), &src.zv.front(), src.n, myidstart, src.myidstart);
    }
}

void Particles::update_stage1(const float dt, const int timestepid)
	{
	    FILE * fdiag = fopen("diag-equilibrate.txt", timestepid > 0 ? "a" : "w");
	    
	    if (timestepid % steps_per_dump == 0)
	    {
		printf("step %d\n", timestepid);
		float t = timestepid * dt;
		diag(fdiag, t);
		diag(stdout, t);
	    }
	    
	    fclose(fdiag);
		
	    _up(xv, xa, dt * 0.5);
	    _up(yv, ya, dt * 0.5);
	    _up(zv, za, dt * 0.5);

	    _up_enforce(xp, xv, dt);
	    _up_enforce(yp, yv, dt);
	    _up_enforce(zp, zv, dt);

	    if (bouncer != nullptr)
		bouncer->bounce(*this, dt);
	};
void Particles::equilibrate(const float kBT, const double tend, const double dt)
{
    internal_forces(kBT, dt);

    vmd_xyz("ic.xyz", false);

    const size_t nt = (int)(tend / dt);

    for(int it = 0; it < nt; ++it)
    {
	update_stage1(dt, it);

	if (bouncer != nullptr)
	    bouncer->bounce(*this, dt);
		
	internal_forces(kBT, dt);

	update_stage2(dt, it);
    }
}

void Bouncer::compute_forces(const float kBT, const double dt, Particles& freeParticles) 
{
    freeParticles.internal_forces_bipartite(kBT, dt,
					&frozen.xp.front(), &frozen.yp.front(), &frozen.zp.front(),
					&frozen.xv.front(), &frozen.yv.front(), &frozen.zv.front(),
					frozen.n, frozen.myidstart, frozen.myidstart);
}

struct SandwichBouncer: Bouncer
{
    float half_width = 1;
    
    SandwichBouncer( const float L):
	Bouncer(L) { }

    bool _handle_collision(float& x, float& y, float& z,
			   float& u, float& v, float& w,
			   float& dt)
	{
	    if (fabs(z) - half_width <= 0)
		return false;
	    
	    const float xold = x - dt * u;
	    const float yold = y - dt * v;
	    const float zold = z - dt * w;

	    assert(fabs(zold) - half_width <= 0);
	    assert(fabs(w) > 0);

	    const float s = 1 - 2 * signbit(w);
	    const float t = (s * half_width - zold) / w;

	    assert(t >= 0);
	    assert(t <= dt);
		    
	    const float lambda = 2 * t - dt;
		    
	    x = xold + lambda * u;
	    y = yold + lambda * v;
	    z = zold + lambda * w;
	    
	    assert(fabs(zold + lambda * w) - half_width <= 0);

	    u = -u;
	    v = -v;
	    w = -w;
	    dt = dt - t;

	    return true;
	}
    
    void bounce(Particles& dest, const float _dt)
	{
	    for(int i = 0; i < dest.n; ++i)
	    {
		float dt = _dt;
		float x = dest.xp[i];
		float y = dest.yp[i];
		float z = dest.zp[i];
		float u = dest.xv[i];
		float v = dest.yv[i];
		float w = dest.zv[i];
		
		if ( _handle_collision(x, y, z, u, v, w, dt) )
		{
		    dest.xp[i] = x;
		    dest.yp[i] = y;
		    dest.zp[i] = z;
		    dest.xv[i] = u;
		    dest.yv[i] = v;
		    dest.zv[i] = w;
		}
	    }
	}

    void _mark(bool * const freeze, Particles p)
	{
	    for(int i = 0; i < p.n; ++i)
		freeze[i] = !(fabs(p.zp[i]) <= half_width);
	}
};

#if 0
FunnelObstacle kirill(32/3, 40, 128);
#else
struct Kirill
{
    bool isInside(const float x, const float y)
	{
	    const float xc = 0, yc = 0;
	    const float radius2 = 4;
	    
	    const float r2 =
		(x - xc) * (x - xc) +
		(y - yc) * (y - yc) ;

	    return r2 < radius2;
	}
    
} kirill;
#endif
   
/*
struct TomatoSandwich: SandwichBouncer
{
    float xc = 0, yc = 0, zc = 0;
    float radius2 = 1;

    TomatoSandwich(const float L): SandwichBouncer(L) {}

     void _mark(bool * const freeze, Particles p)
	{
	    SandwichBouncer::_mark(freeze, p);

	    for(int i = 0; i < p.n; ++i)
	    {
		const float x = p.xp[i] - xc;
		const float y = p.yp[i] - yc;
#if 0
		freeze[i] |= kirill.isInside(x, y);
#else
		const float r2 = x * x + y * y;

		freeze[i] |= r2 < radius2;
#endif
	    }
	}
 
    float _compute_collision_time(const float _x0, const float _y0,
			  const float u, const float v, 
			  const float xc, const float yc, const float r2)
	{
	    const float x0 = _x0 - xc;
	    const float y0 = _y0 - yc;
	    	    
	    const float c = x0 * x0 + y0 * y0 - r2;
	    const float b = 2 * (x0 * u + y0 * v);
	    const float a = u * u + v * v;
	    const float d = sqrt(b * b - 4 * a * c);

	    return (-b - d) / (2 * a);
	}

    bool _handle_collision(float& x, float& y, float& z,
			   float& u, float& v, float& w,
			   float& dt)
	{
#if 0
	    if (!kirill.isInside(x, y))
		return false;

	    const float xold = x - dt * u;
	    const float yold = y - dt * v;
	    const float zold = z - dt * w;

	    float t = 0;
	    
	    for(int i = 1; i < 30; ++i)
	    {
		const float tcandidate = t + dt / (1 << i);
		const float xcandidate = xold + tcandidate * u;
		const float ycandidate = yold + tcandidate * v;
		
		 if (!kirill.isInside(xcandidate, ycandidate))
		     t = tcandidate;
	    }

	    const float lambda = 2 * t - dt;
		    
	    x = xold + lambda * u;
	    y = yold + lambda * v;
	    z = zold + lambda * w;
	   
	    u  = -u;
	    v  = -v;
	    w  = -w;
	    dt = dt - t;

	    return true;
	    
#else
	    const float r2 =
		(x - xc) * (x - xc) +
		(y - yc) * (y - yc) ;
		
	    if (r2 >= radius2)
		return false;
	    
	    assert(dt > 0);
			
	    const float xold = x - dt * u;
	    const float yold = y - dt * v;
	    const float zold = z - dt * w;

	     const float r2old =
		(xold - xc) * (xold - xc) +
		 (yold - yc) * (yold - yc) ;

	     if (r2old < radius2)
		 printf("r2old : %.30f\n", r2old);
	     
	    assert(r2old >= radius2);

	    const float t = _compute_collision_time(xold, yold, u, v, xc, yc, radius2);
	    if (t < 0)
		printf("t is %.20e\n", t);
	    
	    assert(t >= 0);
	    assert(t <= dt);
		    
	    const float lambda = 2 * t - dt;
		    
	    x = xold + lambda * u;
	    y = yold + lambda * v;
	    z = zold + lambda * w;
	    
	    u  = -u;
	    v  = -v;
	    w  = -w;
	    dt = dt - t;

	    return true;
#endif
	}
    
    void bounce(Particles& dest, const float _dt)
	{
	    int gfailcc = 0, gokcc = 0;
	    
#pragma omp parallel
	    {
		int failcc = 0, okcc = 0;

#pragma omp for
		for(int i = 0; i < dest.n; ++i)
		{
		    float x = dest.xp[i];
		    float y = dest.yp[i];
		    float z = dest.zp[i];
		    float u = dest.xv[i];
		    float v = dest.yv[i];
		    float w = dest.zv[i];
		    float dt = _dt;
		    
		    bool wascolliding = false, collision;
		    int passes = 0;
		    do
		    {
			collision = false;
			collision |= SandwichBouncer::_handle_collision(x, y, z, u, v, w, dt);
			collision |= _handle_collision(x, y, z, u, v, w, dt);
		    
			wascolliding |= collision;
			passes++;

			if (passes >= 100)
			    break;
		    }
		    while(collision);

		    if (passes >= 2)
			if (!collision)
			    okcc++;//
			else
			    failcc++;//
		
		    if (wascolliding)
		    {
			dest.xp[i] = x;
			dest.yp[i] = y;
			dest.zp[i] = z;
			dest.xv[i] = u;
			dest.yv[i] = v;
			dest.zv[i] = w;
		    }
		}

#pragma omp critical
		{
		    gfailcc += failcc;
		    gokcc += okcc;
		}
	    }

	    if (gokcc)
		printf("successfully solved %d complex collisions\n", gokcc);

	    if (gfailcc)
	    {
		printf("FAILED to solve %d complex collisions\n", gfailcc);
		abort();
	    }
	}
};
*/

struct ParticlesSEM : Particles
{
    using Particles::Particles;//(const int n, const float L);

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
				    n, params.rcutoff, L, L, L, params.gamma, kBT, dt, params.u0, params.rho, params.req, params.D, params.rc);
	
	    if (bouncer != nullptr)
	    {
		Particles& src = bouncer->frozen;
	
		internal_forces_bipartite(kBT, dt,
				      &src.xp.front(), &src.yp.front(), &src.zp.front(),
				      &src.xv.front(), &src.yv.front(), &src.zv.front(), src.n, myidstart, src.myidstart);
	    }
	}
};

void interforces(Particles& dpdp, ParticlesSEM& semp, double kBT, double dt)
{
    const float L = dpdp.L;
    
    const float xdomainsize = L;
    const float ydomainsize = L;
    const float zdomainsize = L;

    const float invrc = 1.;
    const float gamma = 45;
    const float sigma = sqrt(2 * gamma * kBT);
    const float sigmaf = sigma / sqrt(dt);
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
			      sigma,  1 / sqrt(dt));
}

void microfluidics_simulation(Particles& dpdp, ParticlesSEM& semp, const float kBT, const double tend, const double dt)
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

	interforces(dpdp, semp, kBT, dt);

	dpdp.update_stage2(dt, it);
	semp.update_stage2(dt, it);
    }
}

#include "../vanilla-walls/funnel-bouncer.h"

int main()
{
    //abort();
    const float L = 40;
    const int Nm = 3;
    const int n = L * L * L * Nm;
    
    Particles particles(n, L);
    particles.equilibrate(.1, 10, 0.02);

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
    
    remaining.bouncer = &bouncer;
    remaining.yg = 0.01;
    remaining.steps_per_dump = 5;

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
    cell.bouncer = &bouncer;
    cell.yg = 0.00;
    cell.steps_per_dump = 5;
    //cell.equilibrate(.1, 30*3*5, 0.02);
    //exit(0);


    Particles sys(0, L);
    sys.name = "solvent";
    sys.bouncer = &bouncer;
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

     microfluidics_simulation(sys, cell, .1, 30*3*5, 0.02);
    //sys.equilibrate(.1, 30*3*5, 0.02);
}

    
