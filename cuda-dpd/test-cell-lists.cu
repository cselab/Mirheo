/*
 *  test-cell-lists.cu
 *  Part of uDeviceX/cuda-dpd-sem/
 *
 *  Created and authored by Diego Rossinelli on 2014-08-08.
 *  Copyright 2015. All rights reserved.
 *
 *  Users are NOT authorized
 *  to employ the present software for their own publications
 *  before getting a written permission from the author of this file.
 */

#include <cstdio>
#include <cassert>
#include <utility>

void build_clists(float * const device_xyzuvw, int np, const float rc, 
		  const int xcells, const int ycells, const int zcells,
		  const float xdomainstart, const float ydomainstart, const float zdomainstart,
		  int * const host_order, int * device_cellsstart, int * device_cellscount,
		  std::pair<int, int *> * nonemptycells = NULL);
#include <set>

#include <thrust/device_vector.h>

using namespace thrust;

template<typename T>
T * _ptr(device_vector<T>& v)
{
    return raw_pointer_cast(&v[0]);
}

extern device_vector<int> loffsets, yzcid, outid, yzhisto, dyzscan;

struct FailureTest
{
    int bufsize;
    int * maxstripe, * dmaxstripe;
} extern failuretest;

extern bool clists_perfmon;

void myfill(device_vector<float>& d, const double ext)
{
    const int N = d.size();
    host_vector<float> v(N);
    printf("N is %d\n", N);
    for(int i = 0; i < N; ++i)
	v[i] = -0.5 * ext + drand48() * ext;

    copy(v.begin(), v.end(), d.begin());
}

void test(const int L)
{
    clists_perfmon = true;
    
    const int XL = L;
    const int YL = L;
    const int ZL = L;

    const float densitynumber = 3;
    const int N = densitynumber * XL * YL * ZL;
    const float rc = 1;
    
    device_vector<float> xp(N), yp(N), zp(N);
    device_vector<float> xv(N), yv(N), zv(N);
    
    myfill(xp, XL);
    myfill(yp, YL);
    myfill(zp, ZL);

    myfill(xv, 1);
    myfill(yv, 1);
    myfill(zv, 1);

    //best case scenario
    //if (false)
    {
	host_vector<float> x(N), y(N), z(N);
	
	for(int i = 0; i < N; ++i)
	{
	    const int cid = i / 3;
	    const int xcid = cid % XL;
	    const int ycid = (cid / XL) % YL;
	    const int zcid = cid / XL / YL;

	    x [i] = -0.5 * XL + max(0.f, min((float)XL - 0.1, xcid + 0.5 + 3 * (drand48() - 0.5)));
	    y [i] = -0.5 * YL + max(0.f, min((float)YL - 0.1, ycid + 0.5 + 3 * (drand48() - 0.5)));
	    z [i] = -0.5 * ZL + max(0.f, min((float)ZL - 0.1, zcid + 0.5 + 3 * (drand48() - 0.5)));
	}

	xp = x;
	yp = y;
	zp = z;
    }
    
    printf("my fill is done\n");
    
    int3 ncells = make_int3((int)ceil(XL / rc), (int)ceil(YL / rc), (int)ceil(ZL/rc));
    float3 domainstart = make_float3(-0.5 * XL, - 0.5 * YL, - 0.5 * ZL);
    
    const int ntotcells = ncells.x * ncells.y * ncells.z;
    device_vector<int> start(ntotcells), count(ntotcells);
    device_vector<float> particles_aos(N * 6);
    {
	host_vector<float> _xp(xp), _yp(yp), _zp(zp), _xv(xv), _yv(yv), _zv(zv), _particles_aos(6 * N);
	
	for(int i = 0; i < N; ++i)
	{
	    _particles_aos[0 + 6 * i] = _xp[i];
	    _particles_aos[1 + 6 * i] = _yp[i];
	    _particles_aos[2 + 6 * i] = _zp[i];
	    _particles_aos[3 + 6 * i] = _xv[i];
	    _particles_aos[4 + 6 * i] = _yv[i];
	    _particles_aos[5 + 6 * i] = _zv[i];
	}
	
	particles_aos = _particles_aos;
    }

    host_vector<int> order(N);
    
    build_clists(_ptr(particles_aos), N,  rc, 
		 ncells.x, ncells.y, ncells.z,
		 -0.5 * XL, -0.5 * YL, -0.5 * ZL,
		 &order.front(), _ptr(start), _ptr(count), NULL);
    
#ifndef NDEBUG
    cudaThreadSynchronize();
    {
	host_vector<float> y = yp, z = zp;
	
	host_vector<int> yzhist(ncells.y * ncells.z);
	
	for(int i = 0; i < N; ++i)
	{
	    int ycid = (int)(floor(y[i] - domainstart.y) / rc);
	    int zcid = (int)(floor(z[i] - domainstart.z) / rc);

	    const int entry = ycid + ncells.y * zcid;
	    yzhist[entry]++;
	}

	std::set<int> subids[ncells.y * ncells.z];
	
	//printf("reading global histo: \n");
	int s = 0;
	for(int i = 0; i < yzhisto.size(); ++i)
	{
	    // printf("%d reading %d ref is %d\n", i, (int)yzhisto[i], (int)yzhist[i]);
	    assert(yzhisto[i]  == yzhist[i]);
	    s += yzhisto[i];

	    for(int k = 0; k < yzhist[i]; ++k)
		subids[i].insert(k);
	}
	//printf("s == %d is equal to %d == N\n", s , N);
	assert(s == N);
	
	for(int i = 0; i < N; ++i)
	{
	    int ycid = (int)(floor(y[i] - domainstart.y) / rc);
	    int zcid = (int)(floor(z[i] - domainstart.z) / rc);

	    const int entry = ycid + ncells.y * zcid;

	    const int loff = loffsets[i];
	    const int en = yzcid[i];

	    assert(en == entry);

	    assert(subids[en].find(loff) != subids[en].end());
	    subids[en].erase(loff);
	}

	for(int i = 0; i < yzhisto.size(); ++i)
	    assert(subids[i].size() == 0);

	printf("first level   verifications passed.\n");

	const int mymax = *max_element(yzhist.begin(), yzhist.end());
	printf("mymax: %d maxstripe: %d\n", mymax, *failuretest.maxstripe);
	assert(mymax == *failuretest.maxstripe);
	//assert(false);
    }

    {
	int s = 0;
	for(int i = 0; i < dyzscan.size(); ++i)
	{
	    //printf("%d -> %d (%d)\n", i, (int)dyzscan[i], (int) yzhisto[i]);
	    assert(dyzscan[i] == s);
	    s += yzhisto[i];
	}
    }

    {
	host_vector<int> lut = dyzscan;
	
	for(int i = 0; i < N; ++i)
	{
	    const int landing = outid[i];
	    	    
	    const int entry = yzcid[landing];
	    const int base = lut[entry];
	    const int offset = loffsets[landing];
	    //printf("%d: %d -> %d\n", i, landing, base + offset);
	    assert(i == base + offset);
	}
  
	printf("second level   verification passed\n"); 
    }

    {
	host_vector<int> s(start), c(count);
	host_vector<float> aos(particles_aos);
	host_vector<bool> marked(N);

	//printf("start[0] : %d\n", (int)start[0]);
	assert(start[0] == 0);
	
	for(int iz = 0; iz < ZL; ++iz)
	    for(int iy = 0; iy < YL; ++iy)
		for(int ix = 0; ix < XL; ++ix)
		{
		    const int cid = ix + XL * (iy + YL * iz);
		    const int mys = s[cid];
		    const int myc = c[cid];

		    //intf("cid %d : my start and count are %d %d\n", cid, mys, myc);
		    assert(mys >= 0 && mys < N);
		    assert(myc >= 0 && myc <= N);

		    for(int i = mys; i < mys + myc; ++i)
		    {
			assert(!marked[i]);
			const float x = aos[0 + 6 * i];
			const float y = aos[1 + 6 * i];
			const float z = aos[2 + 6 * i];
			
			const float xcheck = x - domainstart.x;
			const float ycheck = y - domainstart.y;
			const float zcheck = z - domainstart.z;

			//printf("checking p %d: %f %f %f  ref: %d %d %d\n", i, xcheck , ycheck, zcheck, ix, iy, iz);
			assert(xcheck >= ix && xcheck < ix + 1);
			assert(ycheck >= iy && ycheck < iy + 1);
			assert(zcheck >= iz && zcheck < iz + 1);
						
			marked[i] = true;
		    }
		}

	printf("third-level verification passed.\n");
    }

    {
	std::set<int> ids;
	for(int i = 0; i < N; ++i)
	{
	    const int key = order[i];
	    //printf("%d : %d\n", i, key);
	    assert(key >= 0);
	    assert(key < N);
	    assert(ids.find(key) == ids.end());
	    ids.insert(key);
	}
	
    }

     printf("test is done\n");
#endif

    
}

int main()
{
    for(int i = 0; i < 10; ++i)
    {
	if (i < 5)
	    test(40);
	else
	    test(20);
    }
    
    return 0;
}