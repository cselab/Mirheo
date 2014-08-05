/*
 * rbc_ring.cpp
 *
 *  Created on: Jul 16, 2014
 *      Author: kirill lykov
 */
#include <iostream>
#include <cstdlib>
#include <cmath>
#include <cstdio>
#include <cassert>

#include <algorithm>
#include <vector>
#include <cmath>

// cuda headers
#include <thrust/reduce.h>
#include <thrust/sequence.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

typedef float real;
typedef unsigned int sizeType;

typedef thrust::host_vector<real> hvector;
typedef thrust::device_vector<real> dvector;

// global variables
const real boxLength = 5.0;

// dpd code works only if natoms = 2^N
const size_t nrings = 1;
const size_t natomsPerRing = 10;
const size_t nfluidAtoms = 0; // boxLength * boxLength * boxLength; // density 1
const size_t natoms = nrings * natomsPerRing + nfluidAtoms;

hvector xp(natoms), yp(natoms), zp(natoms),
        xv(natoms), yv(natoms), zv(natoms),
        xa(natoms), ya(natoms), za(natoms);

// dpd parameters
const real dtime = 0.001;
const real dtinvsqrt = 31.6227766017; //1.0 / sqrtf(dtime);
const real kbT = 0.1;
const size_t timeEnd = 100;

const real a0 = 500.0, gamma0 = 4.5, cut = 1.2, cutsq = cut * cut, kPower = 0.25,
    sigma = 0.9486; //sqrt(2.0 * kbT * gamma0);

// WLC bond parameters (assumed DPD length unit is 0.5*real)
const real lambda = 2.5e-4;
const real lmax  = 1.3;

// bending angle parameters
const real kbend = 50.0 * kbT;
const real theta = M_PI - 2.0 * M_PI / natomsPerRing;

// misc parameters
const size_t outEvery = 1;
const real ringRadius = 1.0;

#ifdef NEWTONIAN
#include <random>
std::random_device rd;
std::mt19937 gen(rd());
std::normal_distribution<> dgauss(0, 1);

real getGRand(size_t, size_t, size_t)
{
  return dgauss(gen);
}
#else
__device__ __host__ real saru(unsigned int seed1, unsigned int seed2, unsigned int seed3)
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

    real res = r / (4294967295.0);
    return res;
}

// don't know how to have one routine for both with min/max
__host__ real h_getGRand(size_t i, size_t j, size_t idtimestep)
{
  const real mysaru = saru(std::min(i, j), std::max(i, j), idtimestep);
  return 3.464101615 * mysaru - 1.732050807;
}


__device__ real d_getGRand(size_t i, size_t j, size_t idtimestep)
{
  const real mysaru = saru(min((unsigned int)i, (unsigned int)j), max((unsigned int)i, (unsigned int)j), idtimestep);
  return 3.464101615 * mysaru - 1.732050807;
}
#endif

// **** aux routines ******
// might be opened by OVITO and xmovie
void lammps_dump(const char* path, real* xs, real* ys, real* zs, const size_t natoms, size_t timestep, real boxLength)
{
  bool append = timestep > 0;
  FILE * f = fopen(path, append ? "a" : "w");

  if (f == NULL)
  {
    std::cout << "I could not open the file " << path << "Aborting now" << std::endl;
    abort();
  }

  // header
  fprintf(f, "ITEM: TIMESTEP\n%lu\n", timestep);
  fprintf(f, "ITEM: NUMBER OF ATOMS\n%lu\n", natoms);
  fprintf(f, "ITEM: BOX BOUNDS pp pp pp\n%g %g\n%g %g\n%g %g\n",
      -boxLength/2.0, boxLength/2.0, -boxLength/2.0, boxLength/2.0, -boxLength/2.0, boxLength/2.0);

  fprintf(f, "ITEM: ATOMS id type xs ys zs\n");

  // positions <ID> <type> <x> <y> <z>
  // free particles have type 2, while rings 1
  for (size_t i = 0; i < natoms; ++i) {
    int type = i > nrings * natomsPerRing ? 2 : 1;
    fprintf(f, "%lu %d %g %g %g\n", i, type, xs[i], ys[i], zs[i]);
  }

  fclose(f);
}

void dump_force(const char* path,  const hvector& xs, const hvector& ys,
                const hvector& zs, const int n, bool append)
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
      fprintf(f, "%d %f %f %f\n", i, xs[i], ys[i], zs[i]);

    fclose(f);

    printf("vmd_xyz: wrote to <%s>\n", path);
}

real innerProd(const real* v1, const real* v2)
{
  return v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2];
}

real norm2(const real* v)
{
  return v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
}

struct SaxpyOp {
  const real m_coeff;
  SaxpyOp(real coeff) : m_coeff(coeff) {}
  __host__ __device__ real operator()(const real& x, const real& y) const
  {
    return m_coeff * x + y;
  }
};

// delta is difference between coordinates of particles in a bond
void minImage(real* delta)
{
  for (size_t i = 0; i < 3; ++i)
    if (fabs(delta[i]) > 0.5 * boxLength) {
      if (delta[i] < 0.0) delta[i] += boxLength;
      else delta[i] -= boxLength;
    }
}

// set up coordinates
void getRandPoint(real& x, real& y, real& z)
{
   x = drand48() * boxLength - boxLength/2.0;
   y = drand48() * boxLength - boxLength/2.0;
   z = drand48() * boxLength - boxLength/2.0;
}

bool areEqual(const real& left, const real& right)
{
    const real tolerance = 1e-2;
    return fabs(left - right) < tolerance;
}

// **** initialization *****
void addRing(size_t indRing)
{
  real cmass[3];
  getRandPoint(cmass[0], cmass[1], cmass[2]);

  for (size_t indLocal = 0; indLocal < natomsPerRing; ++indLocal) {
    size_t i = natomsPerRing * indRing + indLocal;
    real angle = 2.0 * M_PI / natomsPerRing * i;
    xp[i] = ringRadius * cos(angle) + cmass[0];
    yp[i] = ringRadius * sin(angle) + cmass[1];
    zp[i] = cmass[2];
  }
}

void initPositions()
{
  for (size_t indRing = 0; indRing < nrings; ++indRing) {
    addRing(indRing);
  }

  for (size_t i = nrings * natomsPerRing; i < natoms; ++i) {
    getRandPoint(xp[i], yp[i], zp[i]);
  }
}

// forces computations splitted by the type
void calcDpdForces(size_t timeStep)
{
  real dtinvsqrt = 1.0 / sqrt(dtime);
  for (size_t i = 0; i < natoms; ++i)
  {
#ifdef NEWTONIAN
    for (size_t j = i + 1; j < natoms; ++j)
#else
    for (size_t j = 0; j < natoms; ++j)
#endif
    {
      if (i == j) continue;
      real del[] = {xp[i] - xp[j], yp[i] - yp[j], zp[i] - zp[j]};
      minImage(del);

      real rsq = norm2(del);
      if (rsq < cutsq)
      {
        real r = sqrt(rsq);
        real rinv = 1.0 / r;
        real delv[] = {xv[i] - xv[j], yv[i] - yv[j], zv[i] - zv[j]};

        real dot = innerProd(del, delv);
        real randnum = h_getGRand(i, j, timeStep);

        // conservative force = a0 * wd
        // drag force = -gamma * wd^2 * (delx dot delv) / r
        // random force = sigma * wd * rnd * dtinvsqrt;
        real wd = pow(1.0 - r/cut, kPower);
        double fpair = a0 * (1.0 - r/cut);
        fpair -= gamma0 * wd * wd * dot * rinv;
        fpair += sigma * wd * randnum * dtinvsqrt;
        fpair *= rinv;

        // finally modify forces
        xa[i] += del[0] * fpair;
        ya[i] += del[1] * fpair;
        za[i] += del[2] * fpair;

#ifdef NEWTONIAN
        xa[j] -= del[0] * fpair;
        ya[j] -= del[1] * fpair;
        za[j] -= del[2] * fpair;
#endif
      }
    }
  }
}

// f_wlc(x) = -0.25KbT/p*((1 - x)^-2 + 4x - 1),
// where x := rij/l_max, p is persistent length
// if we assume that water is always after rings
// than it will work with solvent
void calcBondForcesWLC()
{
  for (size_t indRing = 0; indRing < nrings; ++indRing)
  {
    for (size_t indLocal = 0; indLocal < natomsPerRing; ++indLocal)
    {
      size_t i1 = natomsPerRing * indRing + indLocal;
      size_t i2 = natomsPerRing * indRing + (indLocal + 1) % natomsPerRing;
      real del[] = {xp[i1] - xp[i2], yp[i1] - yp[i2], zp[i1] - zp[i2]};
      minImage(del);

      real rsq = norm2(del);
      real lsq = lmax * lmax;
      if (rsq > lsq) { //0.9025 is from the FNS_SFO_2006
        //std::cerr << "WORM bond too long: " << timestep << " " << sqrt(rsq) << std::endl;
        assert(false); // debug me
      }

      real rdl = sqrt(rsq / lsq); //rij/l

      real fbond = 1.0 / ( (1.0 - rdl) * (1.0 - rdl) ) + 4.0 * rdl - 1.0;

       //0.25kbT/lambda[..]
      fbond *= -0.25 * kbT / lambda;

      // finally modify forces
      xa[i1] += del[0] * fbond;
      ya[i1] += del[1] * fbond;
      za[i1] += del[2] * fbond;

      xa[i2] -= del[0] * fbond;
      ya[i2] -= del[1] * fbond;
      za[i2] -= del[2] * fbond;
    }
  }
}

void calcAngleForcesBend()
{
  for (size_t indRing = 0; indRing < nrings; ++indRing)
  {
    for (size_t indLocal = 0; indLocal < natomsPerRing; ++indLocal)
    {
      size_t i1 = natomsPerRing * indRing + indLocal;
      size_t i2 = natomsPerRing * indRing + (indLocal + 1) % natomsPerRing;
      size_t i3 = natomsPerRing * indRing + (indLocal + 2) % natomsPerRing;

      // 1st bond
      real del1[] = {xp[i1] - xp[i2], yp[i1] - yp[i2], zp[i1] - zp[i2]};
      minImage(del1);
      real rsq1 = norm2(del1);
      real r1 = sqrt(rsq1);

      // 2nd bond
      real del2[] = {xp[i3] - xp[i2], yp[i3] - yp[i2], zp[i3] - zp[i2]};
      minImage(del2);
      real rsq2 = norm2(del2);
      real r2 = sqrt(rsq2);

      // c = cosine of angle
      real c = del1[0] * del2[0] + del1[1] * del2[1] + del1[2] * del2[2];
      c /= r1 * r2;
      if (c > 1.0) c = 1.0;
      if (c < -1.0) c = -1.0;
      c *= -1.0;

      real a11 = kbend * c / rsq1;
      real a12 = -kbend / (r1 * r2);
      real a22 = kbend * c / rsq2;

      real f1[] = {a11 * del1[0] + a12 * del2[0], a11 * del1[1] + a12 * del2[1], a11 * del1[2] + a12 * del2[2]};
      real f3[] = {a22 * del2[0] + a12 * del1[0], a22 * del2[1] + a12 * del1[1], a22 * del2[2] + a12 * del1[2]};

      // apply force to each of 3 atoms
      xa[i1] += f1[0];
      ya[i1] += f1[1];
      za[i1] += f1[2];

      xa[i2] -= f1[0] + f3[0];
      ya[i2] -= f1[1] + f3[1];
      za[i2] -= f1[2] + f3[2];

      xa[i3] += f3[0];
      ya[i3] += f3[1];
      za[i3] += f3[2];
    }
  }
}

void addStretchForce()
{
  real externalForce = 250.0;
  xa[0] += externalForce;
  xa[5] -= externalForce;
}

void addDrivingForce()
{
  //real drivingForceY = 100.0;
  //std::for_each(ya.begin(), ya.end(), [&](real& in) { in += drivingForceY; });
}

void computeForces(size_t timeStep)
{
  thrust::fill(xa.begin(), xa.end(), 0.0);
  thrust::fill(ya.begin(), ya.end(), 0.0);
  thrust::fill(za.begin(), za.end(), 0.0);

  calcDpdForces(timeStep);
  calcBondForcesWLC();
  calcAngleForcesBend();

  //addStretchForce();
  addDrivingForce();
}

// ********************** CUDA code **************************
// NumBlocks is Nrings, NThreadsPerblock == NparticlesPerRing

#define GET_RAW(v) thrust::raw_pointer_cast(&v[0])

inline __device__ float d_dot(float* a, float* b)
{
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

inline __device__ float d_length(float* v)
{
    return sqrtf(d_dot(v, v));
}

__device__ void d_minImage(float* delta, float boxLength)
{
  for (size_t i = 0; i < 3; ++i)
    if (fabs(delta[i]) > 0.5 * boxLength) {
      if (delta[i] < 0.0) delta[i] += boxLength;
      else delta[i] -= boxLength;
    }
}

__global__ void kernelBonds(const float* d_xp, const float* d_yp, const float* d_zp, float* d_xa, float* d_ya, float* d_za, int natoms)
{
  extern __shared__ float3 s_df[];
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  size_t indRing = blockIdx.x;
  size_t indLocal = threadIdx.x;

  if (tid > natoms) return;

  size_t i1 = natomsPerRing * indRing + indLocal; //tid
  size_t i2 = natomsPerRing * indRing + (indLocal + 1) % natomsPerRing;
  float del[] = {d_xp[i1] - d_xp[i2], d_yp[i1] - d_yp[i2], d_zp[i1] - d_zp[i2]};
  d_minImage(del, boxLength);

  real rsq = d_dot(del, del);
  real lsq = lmax * lmax;
  if (rsq > lsq) { //0.9025 is from the FNS_SFO_2006
    //std::cerr << "WORM bond too long: " << timestep << " " << sqrt(rsq) << std::endl;
    // TODO how to call assert from CUDA?
    //assert(false); // debug me
  }

  float rdl = sqrtf(rsq / lsq); //rij/l

  float fbond = 1.0 / ( (1.0 - rdl) * (1.0 - rdl) ) + 4.0 * rdl - 1.0;

   //0.25kbT/lambda[..]
  fbond *= -0.25 * kbT / lambda;

  s_df[indLocal] = make_float3(del[0] * fbond, del[1] * fbond, del[2] * fbond);
  __syncthreads();
  
  size_t j = (indLocal + 9) % 10;
  d_xa[i1] += s_df[indLocal].x - s_df[j].x;
  d_ya[i1] += s_df[indLocal].y - s_df[j].y;
  d_za[i1] += s_df[indLocal].z - s_df[j].z;
}

void cuda_calcBondForcesWLC(const dvector& d_xp, const dvector& d_yp, const dvector& d_zp, dvector& d_xa, dvector& d_ya, dvector& d_za)
{
  int nThreadsPerBlock = natomsPerRing;
  int nBlocks = nrings;

  kernelBonds<<< nBlocks, nThreadsPerBlock, sizeof(float3) * nThreadsPerBlock >>>(
      GET_RAW(d_xp), GET_RAW(d_yp), GET_RAW(d_zp),
      GET_RAW(d_xa), GET_RAW(d_ya), GET_RAW(d_za),
      nrings * natomsPerRing);
}

__global__ void kernelAngles(const float* d_xp, const float* d_yp, const float* d_zp, 
                             float* d_xa, float* d_ya, float* d_za, int natoms)
{
  extern __shared__ float3 s_df[];
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  size_t indRing = blockIdx.x;
  size_t indLocal = threadIdx.x;

  if (tid > natoms) return;

  size_t i1 = natomsPerRing * indRing + indLocal;
  size_t i2 = natomsPerRing * indRing + (indLocal + 1) % natomsPerRing;
  size_t i3 = natomsPerRing * indRing + (indLocal + 2) % natomsPerRing;

  // 1st bond
  float del1[] = {d_xp[i1] - d_xp[i2], d_yp[i1] - d_yp[i2], d_zp[i1] - d_zp[i2]};
  d_minImage(del1, boxLength);
  float rsq1 = d_dot(del1, del1);
  float r1 = sqrtf(rsq1);

  // 2nd bond
  float del2[] = {d_xp[i3] - d_xp[i2], d_yp[i3] - d_yp[i2], d_zp[i3] - d_zp[i2]};
  d_minImage(del2, boxLength);
  float rsq2 = d_dot(del2, del2);
  float r2 = sqrtf(rsq2);

  // c = cosine of angle
  float c = del1[0] * del2[0] + del1[1] * del2[1] + del1[2] * del2[2];
  c /= r1 * r2;
  if (c > 1.0) c = 1.0;
  if (c < -1.0) c = -1.0;
  c *= -1.0;

  float a11 = kbend * c / rsq1;
  float a12 = -kbend / (r1 * r2);
  float a22 = kbend * c / rsq2;

  // there are two forces, so one is stored in the first part, another in the second
  s_df[indLocal] = make_float3(a11 * del1[0] + a12 * del2[0], a11 * del1[1] + a12 * del2[1], a11 * del1[2] + a12 * del2[2]);
  s_df[indLocal + blockDim.x] = make_float3(a22 * del2[0] + a12 * del1[0], a22 * del2[1] + a12 * del1[1], a22 * del2[2] + a12 * del1[2]);
  __syncthreads();

  size_t j = (indLocal + 8) % 10, k = (indLocal + 9) % 10;

  d_xa[i1] += s_df[indLocal].x + s_df[ blockDim.x + j].x - (s_df[k].x + s_df[ blockDim.x + k ].x);
  d_ya[i1] += s_df[indLocal].y + s_df[ blockDim.x + j].y - (s_df[k].y + s_df[ blockDim.x + k ].y);
  d_za[i1] += s_df[indLocal].z + s_df[ blockDim.x + j].z - (s_df[k].z + s_df[ blockDim.x + k ].z);
}

void cuda_calcAngleForcesBend(const dvector& d_xp, const dvector& d_yp, const dvector& d_zp, 
                              dvector& d_xa, dvector& d_ya, dvector& d_za)
{
  int nThreadsPerBlock = natomsPerRing;
  int nBlocks = nrings;

  kernelAngles<<< nBlocks, nThreadsPerBlock, 2 * sizeof(float3) * nThreadsPerBlock >>>(
      GET_RAW(d_xp), GET_RAW(d_yp), GET_RAW(d_zp),
      GET_RAW(d_xa), GET_RAW(d_ya), GET_RAW(d_za),
      nrings * natomsPerRing);
}

__device__ void kernelDPDPairCore(sizeType i, sizeType j, sizeType timeStep, 
                                  const float3 pi, const float3 pj, 
                                  const float3 vi, const float3 vj, float* res)
{
  real del[] = {pi.x - pj.x, pi.y - pj.y, pi.z - pj.z};
  d_minImage(del, boxLength);

  real rsq = d_dot(del, del);
  if (rsq < cutsq)
  {
    real r = sqrtf(rsq);
    real rinv = 1.0 / r;
    real delv[] = {vi.x - vj.x, vi.y - vj.y, vi.z - vj.z};

    real dot = d_dot(del, delv);
    real randnum = d_getGRand(i, j, timeStep);

    // conservative force = a0 * wd
    // drag force = -gamma * wd^2 * (delx dot delv) / r
    // random force = sigma * wd * rnd * dtinvsqrt;
    real wd = powf(1.0 - r/cut, kPower);
    real fpair = a0 * (1.0 - r/cut);
    fpair -= gamma0 * wd * wd * dot * rinv;
    fpair += sigma * wd * randnum * dtinvsqrt;
    fpair *= rinv;

    res[0] = del[0] * fpair;
    res[1] = del[1] * fpair;
    res[2] = del[2] * fpair;
  }
}

__global__ void kernelDPDRingInter(const float* d_xp, const float* d_yp, const float* d_zp,
                              const float* d_xv, const float* d_yv, const float* d_zv,
                              float* d_xa, float* d_ya, float* d_za, int natoms, size_t timeStep)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  sizeType i = tid % 10;

  float df[] = {0.0, 0.0, 0.0};

 if (tid > 29) 
  {
    float dfBuf[3];
    float3 pi = make_float3(d_xp[i], d_yp[i], d_zp[i]);
    float3 vi = make_float3(d_xv[i], d_yv[i], d_zv[i]);
    for (sizeType j = tid / 10; j < 10; j += 3)
    {
      float3 pj = make_float3(d_xp[j], d_yp[j], d_zp[j]);
      float3 vj = make_float3(d_xv[j], d_yv[j], d_zv[j]);
      if (i == j || i > 9 || j > 9)
        continue;

      kernelDPDPairCore(i, j, timeStep, pi, pj, vi, vj, dfBuf);
      df[0] += dfBuf[0]; df[1] += dfBuf[1]; df[2] += dfBuf[2];
    }
  }
  sizeType laneid =  threadIdx.x % 32;

  for (sizeType i = 0; i < 3; ++i) {
    df[i] += __shfl(df[i], laneid + 10);
    df[i] += __shfl(df[i], laneid + 20);
  }

  if (laneid < 10) {
    d_xa[i] += df[0];
    d_ya[i] += df[1];
    d_za[i] += df[2];
  }
}

// forces computations splitted by the type
__global__ void kernelDPDpair(const float* d_xp, const float* d_yp, const float* d_zp,
                              const float* d_xv, const float* d_yv, const float* d_zv,
                              float* d_xa, float* d_ya, float* d_za, int natoms, size_t timeStep)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  size_t i = blockIdx.x;
  size_t j = threadIdx.x;

  if (tid > natoms * natoms) return;

  extern __shared__ float3 s_df[];

  float3 pi = make_float3(d_xp[i], d_yp[i], d_zp[i]);
  float3 vi = make_float3(d_xv[i], d_yv[i], d_zv[i]);
  float3 pj = make_float3(d_xp[j], d_yp[j], d_zp[j]);
  float3 vj = make_float3(d_xv[j], d_yv[j], d_zv[j]);  
  float  res[] = {0.0, 0.0, 0.0};
  if (i != j)
    kernelDPDPairCore(i, j, timeStep, pi, pj, vi, vj, res);
  s_df[j].x = res[0]; s_df[j].y = res[1]; s_df[j].z = res[2];
  __syncthreads();

  // f[i] = sum{j}(df[j]);)
  // for reductions, threadsPerBlock must be a power of 2
  int k = blockDim.x/2;
  while (k != 0) {
    if (threadIdx.x < k) {
      s_df[threadIdx.x].x += s_df[threadIdx.x + k].x;
      s_df[threadIdx.x].y += s_df[threadIdx.x + k].y;
      s_df[threadIdx.x].z += s_df[threadIdx.x + k].z;
    }
    __syncthreads();
    k /= 2;
  }

  if (threadIdx.x == 0) {
    d_xa[i] += s_df[0].x;
    d_ya[i] += s_df[0].y;
    d_za[i] += s_df[0].z;
  }
}

void cuda_calcDpdForces(const dvector& d_xp, const dvector& d_yp, const dvector& d_zp,
                        const dvector& d_xv, const dvector& d_yv, const dvector& d_zv,
                        dvector& d_xa, dvector& d_ya, dvector& d_za, size_t timeStep)
{
  // TODO natoms per block - don't think it is a good idea, just to try
  int nThreadsPerBlock = natoms;
  int nBlocks = natoms;

  kernelDPDRingInter<<< nBlocks, nThreadsPerBlock >>>(
      GET_RAW(d_xp), GET_RAW(d_yp), GET_RAW(d_zp),
      GET_RAW(d_xv), GET_RAW(d_yv), GET_RAW(d_zv),
      GET_RAW(d_xa), GET_RAW(d_ya), GET_RAW(d_za),
      natoms, timeStep);
}

void cuda_computeForcesCheck(size_t timeStep)
{
  dvector d_xa(natoms), d_ya(natoms), d_za(natoms),
          d_xp(natoms), d_yp(natoms), d_zp(natoms),
          d_xv(natoms), d_yv(natoms), d_zv(natoms);

  thrust::fill(d_xa.begin(), d_xa.end(), 0.0);
  thrust::fill(d_ya.begin(), d_ya.end(), 0.0);
  thrust::fill(d_za.begin(), d_za.end(), 0.0);

  d_xp = xp; d_yp = yp; d_zp = zp;
  d_xv = xv; d_yv = yv; d_zv = zv;

  cuda_calcDpdForces(d_xp, d_yp, d_zp, d_xv, d_yv, d_zv,
      d_xa, d_ya, d_za, timeStep);
  cudaDeviceSynchronize();
  //cuda_calcBondForcesWLC(d_xp, d_yp, d_zp, d_xa, d_ya, d_za);
  //cudaDeviceSynchronize();  
  //cuda_calcAngleForcesBend(d_xp, d_yp, d_zp, d_xa, d_ya, d_za);
  //cudaDeviceSynchronize();
  
  thrust::fill(xa.begin(), xa.end(), 0.0);
  thrust::fill(ya.begin(), ya.end(), 0.0);
  thrust::fill(za.begin(), za.end(), 0.0);
  
  //to check that xa == d_xa; ya == d_ya; za == d_za;  
  calcDpdForces(timeStep);
  //calcBondForcesWLC();
  //calcAngleForcesBend();

  // check that computations are correct
  thrust::host_vector<real> h_xa(d_xa), h_ya(d_ya), h_za(d_za);
  for (size_t i = 0; i < natoms; ++i) {
    if (!areEqual(xa[i], h_xa[i]) ||
        !areEqual(ya[i], h_ya[i])||
        !areEqual(za[i], h_za[i])) {

        dump_force("force-cpu.txt", xa, ya, za, natoms, false);
        dump_force("force-gpu.txt", h_xa, h_ya, h_za, natoms, false);
        std::cout << i << "| diff=" << fabs(xa[i] - h_xa[i]) << " " << fabs(ya[i] - h_ya[i]) 
            << " " << fabs(za[i] - h_za[i]) << std::endl;
        abort();
    }
  }
}

void cuda_computeForces(
    const dvector& d_xp, const dvector& d_yp, const dvector& d_zp,
    const dvector& d_xv, const dvector& d_yv, const dvector& d_zv, 
    dvector& d_xa, dvector& d_ya, dvector& d_za, size_t timeStep)
{
  thrust::fill(d_xa.begin(), d_xa.end(), 0.0);
  thrust::fill(d_ya.begin(), d_ya.end(), 0.0);
  thrust::fill(d_za.begin(), d_za.end(), 0.0);

  cuda_calcDpdForces(d_xp, d_yp, d_zp, d_xv, d_yv, d_zv,
      d_xa, d_ya, d_za, timeStep);
  cudaDeviceSynchronize();
  cuda_calcBondForcesWLC(d_xp, d_yp, d_zp, d_xa, d_ya, d_za);
  cudaDeviceSynchronize();  
  cuda_calcAngleForcesBend(d_xp, d_yp, d_zp, d_xa, d_ya, d_za);
  cudaDeviceSynchronize(); 
}

// initial integration of velocity-verlet
void cuda_initialIntegrate(dvector& d_xp, dvector& d_yp, dvector& d_zp,
                           dvector& d_xv, dvector& d_yv, dvector& d_zv, 
                           const dvector& d_xa, const dvector& d_ya, const dvector& d_za)
{
  thrust::transform(d_xa.begin(), d_xa.end(), d_xv.begin(), d_xv.begin(), SaxpyOp(dtime * 0.5));
  thrust::transform(d_ya.begin(), d_ya.end(), d_yv.begin(), d_yv.begin(), SaxpyOp(dtime * 0.5));
  thrust::transform(d_za.begin(), d_za.end(), d_zv.begin(), d_zv.begin(), SaxpyOp(dtime * 0.5));

  thrust::transform(d_xv.begin(), d_xv.end(), d_xp.begin(), d_xp.begin(), SaxpyOp(dtime));
  thrust::transform(d_yv.begin(), d_yv.end(), d_yp.begin(), d_yp.begin(), SaxpyOp(dtime));
  thrust::transform(d_zv.begin(), d_zv.end(), d_zp.begin(), d_zp.begin(), SaxpyOp(dtime));
}

//final integration of velocity-verlet
void cuda_finalIntegrate(dvector& d_xv, dvector& d_yv, dvector& d_zv, 
                         const dvector& d_xa, const dvector& d_ya, const dvector& d_za)
{
  thrust::transform(d_xa.begin(), d_xa.end(), d_xv.begin(), d_xv.begin(), SaxpyOp(dtime * 0.5));
  thrust::transform(d_ya.begin(), d_ya.end(), d_yv.begin(), d_yv.begin(), SaxpyOp(dtime * 0.5));
  thrust::transform(d_za.begin(), d_za.end(), d_zv.begin(), d_zv.begin(), SaxpyOp(dtime * 0.5));
}

using namespace std;
struct PbcOp {
  __host__ __device__ void operator()(real& coord) const
  {
    real boxlo = -0.5 * boxLength;
    real boxhi = 0.5 * boxLength;
    if (coord < boxlo) {
      coord += boxLength;
    }
    if (coord >= boxhi) {
      coord -= boxLength;
      coord = max(coord, boxlo);
    }
  }
};

void cuda_pbc(dvector& xp, dvector& yp, dvector& zp)
{
  PbcOp op;
  thrust::for_each(xp.begin(), xp.end(), op);
  thrust::for_each(yp.begin(), yp.end(), op);
  thrust::for_each(zp.begin(), zp.end(), op);

  // check that it works
  real boxlo = -0.5 * boxLength;
  real boxhi = 0.5 * boxLength;
  for (size_t i = 0; i < natoms; ++i) {
    if (xp[i] < boxlo || xp[i] >= boxhi ||
        yp[i] < boxlo || yp[i] >= boxhi ||
        zp[i] < boxlo || zp[i] >= boxhi)
      assert(false);
  }
}


// *******************************************************

// initial integration of velocity-verlet
void initialIntegrate()
{
  std::transform(xa.begin(), xa.end(), xv.begin(), xv.begin(), SaxpyOp(dtime * 0.5));
  std::transform(ya.begin(), ya.end(), yv.begin(), yv.begin(), SaxpyOp(dtime * 0.5));
  std::transform(za.begin(), za.end(), zv.begin(), zv.begin(), SaxpyOp(dtime * 0.5));

  std::transform(xv.begin(), xv.end(), xp.begin(), xp.begin(), SaxpyOp(dtime));
  std::transform(yv.begin(), yv.end(), yp.begin(), yp.begin(), SaxpyOp(dtime));
  std::transform(zv.begin(), zv.end(), zp.begin(), zp.begin(), SaxpyOp(dtime));
}

//final integration of velocity-verlet
void finalIntegrate()
{
  std::transform(xa.begin(), xa.end(), xv.begin(), xv.begin(), SaxpyOp(dtime * 0.5));
  std::transform(ya.begin(), ya.end(), yv.begin(), yv.begin(), SaxpyOp(dtime * 0.5));
  std::transform(za.begin(), za.end(), zv.begin(), zv.begin(), SaxpyOp(dtime * 0.5));
}

void pbc()
{
  PbcOp op;
  std::for_each(xp.begin(), xp.end(), op);
  std::for_each(yp.begin(), yp.end(), op);
  std::for_each(zp.begin(), zp.end(), op);

  // check that it works
  real boxlo = -0.5 * boxLength;
  real boxhi = 0.5 * boxLength;
  for (size_t i = 0; i < natoms; ++i) {
    if (xp[i] < boxlo || xp[i] >= boxhi ||
        yp[i] < boxlo || yp[i] >= boxhi ||
        zp[i] < boxlo || zp[i] >= boxhi)
      assert(false);
  }
}

void computeDiams()
{
  real axisal[] = {xp[0] - xp[5], yp[0] - yp[5], zp[0] - zp[5]};
  real daxial = sqrt(norm2(axisal));
  real transverse[] = {0.5 * (xp[2] + xp[3] - xp[7] - xp[8]),
      0.5 * (yp[2] + yp[3] - yp[7] - yp[8]),
      0.5 * (zp[2] + zp[3] - zp[7] - zp[8])};
  real dtrans = sqrt(norm2(transverse));
  std::cout << "Daxial=" << daxial << ", Dtras=" << dtrans << std::endl;
}

int main()
{
  std::cout << "Started computing" << std::endl;
  initPositions();
  FILE * fstat = fopen("diag.txt", "w");

  //dvector d_xp(natoms), d_yp(natoms), d_zp(natoms),
  //        d_xv(natoms), d_yv(natoms), d_zv(natoms),
  //        d_xa(natoms), d_ya(natoms), d_za(natoms);
  
  //d_xp = xp; d_yp = yp; d_zp = zp;
  //d_xv = xv; d_yv = yv; d_zv = zv;
  
  for (size_t timeStep = 0; timeStep < timeEnd; ++timeStep)
  {
    if (timeStep % outEvery == 0)
    {
      std::cout << "t=" << timeStep << std::endl;
      //computeDiams();
      //printStatistics(fstat, timeStep);
    }

    initialIntegrate(/*d_xp, d_yp, d_zp,
                          d_xv, d_yv, d_zv,
                          d_xa, d_ya, d_za*/);
    pbc(/*d_xp, d_yp, d_zp*/);
    if (timeStep % outEvery == 0) {
      //xp = d_xp; yp = d_yp; zp = d_zp;
      lammps_dump("evolution.dump", &xp.front(), &yp.front(), &zp.front(), natoms, timeStep, boxLength);
    }

    cuda_computeForcesCheck(/*d_xp, d_yp, d_zp,
                       d_xv, d_yv, d_zv,
                       d_xa, d_ya, d_za,*/ timeStep);

    finalIntegrate(/*d_xv, d_yv, d_zv,
                        d_xa, d_ya, d_za*/);
  }

  fclose(fstat);
  std::cout << "Ended computing" << std::endl;
  return 0;
}


