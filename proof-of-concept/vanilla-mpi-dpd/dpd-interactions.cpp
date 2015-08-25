/*
 *  dpd-interactions.cpp
 *  Part of CTC/vanilla-mpi-dpd/
 *
 *  Created and authored by Diego Rossinelli on 2014-11-07.
 *  Copyright 2015. All rights reserved.
 *
 *  Users are NOT authorized
 *  to employ the present software for their own publications
 *  before getting a written permission from the author of this file.
 */

#include <algorithm>

#include "dpd-interactions.h"

using namespace std;

#ifdef _WITHCUDA_
#include "cuda-dpd.h"
#endif

//for now saru is at the CORE of our DPD interaction kernel.
//i use and misuse seed1-3 as i see fit, they are NOT always associated to the same things.
float saru(unsigned int seed1, unsigned int seed2, unsigned int seed3)
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

//local interactions deserve a kernel on their own, since they are expected to take most of the computational time.
//saru tag is there to prevent the realization of the same random force twice between different timesteps
void ComputeInteractionsDPD::dpd_kernel(Particle * p, int n, int saru_tag,  Acceleration * a)
{    
#ifdef _WITHCUDA_
    vector<Acceleration> tmp(n);
    vector<int> order(n);
    
    forces_dpd_cuda_aos((float *)p, (float *)&tmp.front(), &order.front(), n, 1, L, L, L, aij, gammadpd, sigma, 1. / sqrt(dt));

    for(int i = 0; i < n; ++i)
	a[order[i]] = tmp[i];

    return;
#endif

    //dummy host implementation
    for(int i = 0; i < n; ++i)
    {
	float xf = 0, yf = 0, zf = 0;

	for(int j = 0; j < n; ++j)
	{
	    if (j == i)
		continue;
	    
	    float _xr = p[i].x[0] - p[j].x[0];
	    float _yr = p[i].x[1] - p[j].x[1];
	    float _zr = p[i].x[2] - p[j].x[2];
       		    
	    float rij2 = _xr * _xr + _yr * _yr + _zr * _zr;
	    float invrij = 1.f / sqrtf(rij2);

	    if (rij2 == 0)
		invrij = 100000;
	    
	    float rij = rij2 * invrij;
	    float wr = max((float)0, 1 - rij);
		    
	    float xr = _xr * invrij;
	    float yr = _yr * invrij;
	    float zr = _zr * invrij;
		  
	    float rdotv = 
		xr * (p[i].u[0] - p[j].u[0]) +
		yr * (p[i].u[1] - p[j].u[1]) +
		zr * (p[i].u[2] - p[j].u[2]);
	    
	    float mysaru = saru(min(i, j), max(i, j), saru_tag);
	    
	    float myrandnr = 3.464101615f * mysaru - 1.732050807f;
		 
	    float strength = (aij - gammadpd * wr * rdotv + sigmaf * myrandnr) * wr;

	    xf += strength * xr;
	    yf += strength * yr; 
	    zf += strength * zr;
	}

	a[i].a[0] = xf;
	a[i].a[1] = yf;
	a[i].a[2] = zf;
    }
}
    
void ComputeInteractionsDPD::dpd_bipartite_kernel(Particle * pdst, int ndst, Particle * psrc, int nsrc,
			  int saru_tag1, int saru_tag2, int saru_mask, Acceleration * a)
{
#ifdef _WITHCUDA_
    if (nsrc > 0 && ndst > 0)
	directforces_dpd_cuda_bipartite((float *)pdst, (float *) a, ndst, (float *)psrc, nsrc, aij, gammadpd, sigma, 1. / sqrt(dt),
					saru_tag1, saru_tag2, saru_mask);

    return ;
#endif
  
    //this will be a CUDA KERNEL in the libcuda-dpd
    for(int i = 0; i < ndst; ++i)
    {
	float xf = 0, yf = 0, zf = 0;

	for(int j = 0; j < nsrc; ++j)
	{
	    float _xr = pdst[i].x[0] - psrc[j].x[0];
	    float _yr = pdst[i].x[1] - psrc[j].x[1];
	    float _zr = pdst[i].x[2] - psrc[j].x[2];
		    
	    float rij2 = _xr * _xr + _yr * _yr + _zr * _zr;
	    float invrij = 1.f / sqrtf(rij2);

	    if (rij2 == 0)
		invrij = 100000;
	    
	    float rij = rij2 * invrij;
	    float wr = max((float)0, 1 - rij);
		    
	    float xr = _xr * invrij;
	    float yr = _yr * invrij;
	    float zr = _zr * invrij;
		
	    float rdotv = 
		xr * (pdst[i].u[0] - psrc[j].u[0]) +
		yr * (pdst[i].u[1] - psrc[j].u[1]) +
		zr * (pdst[i].u[2] - psrc[j].u[2]);
	    
	    float mysaru = saru(saru_tag1, saru_tag2, saru_mask ? i + ndst * j : j + nsrc * i);
	     
	    float myrandnr = 3.464101615f * mysaru - 1.732050807f;
		 
	    float strength = (aij - gammadpd * wr * rdotv + sigmaf * myrandnr) * wr;

	    xf += strength * xr;
	    yf += strength * yr;
	    zf += strength * zr;
	}

	a[i].a[0] = xf;
	a[i].a[1] = yf;
	a[i].a[2] = zf;
    }
}
 
void ComputeInteractionsDPD::dpd_remote_interactions_stage1(Particle * p, int n)
{
    MPI_Status statuses[26];
    
    if (send_pending)
	MPI_CHECK( MPI_Waitall(26, sendreq, statuses) );
	    
    //collect my halo particles into packs.
    //in the non-vanilla version this will be a CUDA KERNEL
    for(int i = 0; i < 26; ++i)
    {
	int d[3] = { (i + 2) % 3 - 1, (i / 3 + 2) % 3 - 1, (i / 9 + 2) % 3 - 1 };

	int halo_start[3], halo_end[3];
	for(int c = 0; c < 3; ++c)
	{
	    halo_start[c] = max(d[c] * L - L/2 - 1, -L/2);
	    halo_end[c] = min(d[c] * L + L/2 + 1, L/2);
	}
	
	for(int j = 0; j < n; ++j)
	{
	    bool halo = true;

	    for(int c = 0; c < 3; ++c)
		halo &= (p[j].x[c] >= halo_start[c] && p[j].x[c] < halo_end[c]);

	    if (halo)
	    {
		mypacks[i].push_back(p[j]);
		myentries[i].push_back(j);
	    }
	}
    }

    // send them to the surrounding ranks.
    for(int i = 0; i < 26; ++i)
    {
	int d[3] = { (i + 2) % 3 - 1, (i / 3 + 2) % 3 - 1, (i / 9 + 2) % 3 - 1 };
	
	int coordsneighbor[3];
	for(int c = 0; c < 3; ++c)
	    coordsneighbor[c] = coords[c] + d[c];

	int dstrank;
	MPI_CHECK( MPI_Cart_rank(cartcomm, coordsneighbor, &dstrank) );
	
	//in the non-vanilla version will use GPUDirect RDMA here
	MPI_CHECK( MPI_Isend(&mypacks[i].front(), mypacks[i].size(), Particle::datatype(), dstrank,
			     tagbase_dpd_remote_interactions + i, cartcomm, sendreq + i) );
    }

    send_pending = true;

    //get remote particle packs from surrounding ranks.
    for(int i = 0; i < 26; ++i)
    {
	int d[3] = { (i + 2) % 3 - 1, (i / 3 + 2) % 3 - 1, (i / 9 + 2) % 3 - 1 };

	int tag = tagbase_dpd_remote_interactions + (2 - d[0]) % 3 + 3 * ((2 - d[1]) % 3 + 3 * ((2 - d[2]) % 3));
	
	MPI_Status status;
	MPI_CHECK( MPI_Probe(MPI_ANY_SOURCE, tag, cartcomm, &status) );

	int count;
	MPI_CHECK( MPI_Get_count(&status, Particle::datatype(), &count) );

	srcpacks[i].resize(count);
	
	//in the non-vanilla version will use GPUDirect RDMA here
	MPI_CHECK( MPI_Irecv(&srcpacks[i].front(), count, Particle::datatype(), MPI_ANY_SOURCE, tag, cartcomm, recvreq + i) );
    }
}
    
void ComputeInteractionsDPD::dpd_remote_interactions_stage2(Particle * p, int n, int saru_tag1, Acceleration * a)
{
    //we want to keep it simple. that's why wait all messages.
    MPI_Status statuses[26];
    MPI_CHECK( MPI_Waitall(26, recvreq, statuses) );
	    
    /* compute saru tags are kind of nightmare to compute:
       we have to make sure to come up with a triple tag which is unique for each interaction in the system
       one tag is given (saru_tag1)
       another one is computed by considering 3d coords of the interacting ranks (saru_tag2[])
       the remaining one is computed inside dpd_bipartite_kernel based on saru_mask[] */
    
    int saru_tag2[26], saru_mask[26];
    for(int i = 0; i < 26; ++i)
    {
	int d[3] = { (i + 2) % 3 - 1, (i / 3 + 2) % 3 - 1, (i / 9 + 2) % 3 - 1 };

	int coordsneighbor[3];
	for(int c = 0; c < 3; ++c)
	    coordsneighbor[c] = (coords[c] + d[c] + dims[c]) % dims[c];

	int indx[3];
	for(int c = 0; c < 3; ++c)
	    indx[c] = min(coords[c], coordsneighbor[c]) * dims[c] + max(coords[c], coordsneighbor[c]);

	saru_tag2[i] = indx[0] + dims[0] * dims[0] * (indx[1] + dims[1] * dims[1] * indx[2]);

	int dstrank;
	MPI_CHECK( MPI_Cart_rank(cartcomm, coordsneighbor, &dstrank) );

	if (dstrank != myrank)
	    saru_mask[i] = min(dstrank, myrank) == myrank;
	else
	{
	    int alter_ego = (2 - d[0]) % 3 + 3 * ((2 - d[1]) % 3 + 3 * ((2 - d[2]) % 3));
	    saru_mask[i] = min(i, alter_ego) == i;
	}
    }

    /* compute interactions with the remote particle packs,
       after properly shifting them to my system of reference
       then update the acceleration vector
       in the non-vanilla version this will be an orchestration of non-blocking CUDA KERNEL calls */
    {
	vector<Acceleration> apacks[26];
	for(int i = 0; i < 26; ++i)
	{
	    int d[3] = { (i + 2) % 3 - 1, (i / 3 + 2) % 3 - 1, (i / 9 + 2) % 3 - 1 };

	    for(int j = 0; j < srcpacks[i].size(); ++j)
		for(int c = 0; c < 3; ++c)
		    srcpacks[i][j].x[c] += d[c] * L;

	    int npack = mypacks[i].size();
	
	    apacks[i].resize(npack);
	
	    dpd_bipartite_kernel(&mypacks[i].front(), npack, &srcpacks[i].front(), srcpacks[i].size(),
				 saru_tag1, saru_tag2[i], saru_mask[i], &apacks[i].front());
	}

	//blend the freshly computed partial results to my local acceleration vector.
	for(int i = 0; i < 26; ++i)
	{
	    Particle * ppack = &mypacks[i].front();
	
	    for(int j = 0; j < mypacks[i].size() ; ++j)
	    {
		int entry = myentries[i][j];

		for(int c = 0; c < 3; ++c)
		    a[entry].a[c] += apacks[i][j].a[c];
	    }
	}
    }

    for(int i = 0; i < 26; ++i)
    {
	mypacks[i].clear();
	myentries[i].clear();
    }
}
