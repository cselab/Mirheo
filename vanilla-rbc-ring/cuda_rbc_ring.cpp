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
//#include <random>
#include <cmath>

// cuda headers
#include <thrust/reduce.h>
#include <thrust/sequence.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

using namespace std;

typedef float real;

typedef thrust::host_vector<real> hvector;
typedef thrust::device_vector<real> dvector;

// global variables
const real boxLength = 10.0;

const size_t nrings = 5;
const size_t natomsPerRing = 10;
const size_t nfluidAtoms = boxLength * boxLength * boxLength; // density 1
const size_t natoms = nrings * natomsPerRing + nfluidAtoms;

hvector xp(natoms), yp(natoms), zp(natoms),
        xv(natoms), yv(natoms), zv(natoms),
        xa(natoms), ya(natoms), za(natoms);

// dpd parameters
const real dtime = 0.001;
const real kbT = 0.1;
const size_t timeEnd = 1000;

const real a0 = 500.0, gamma0 = 4.5, cut = 1.2, cutsq = cut * cut, kPower = 0.25,
    sigma = sqrt(2.0 * kbT * gamma0);

// WLC bond parameters (assumed DPD length unit is 0.5*real)
const real lambda = 2.5e-4;
const real lmax  = 1.3;

// bending angle parameters
const real kbend = 50.0 * kbT;
const real theta = M_PI - 2.0 * M_PI / natomsPerRing;

// misc parameters
const size_t outEvery = 100;
const real ringRadius = 1.0;

// nvcc I use don't know c++11, need something else
//std::random_device rd;
//std::mt19937 gen(rd());
//std::normal_distribution<> dgauss(0, 1);


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

// didn't what to use lambdas because of old version of gdb
// naive integration
void up(hvector& x, hvector& v, real coef)
{
  for (size_t i = 0; i < natoms; ++i)
    x[i] += coef * v[i];
};

void up_enforce(hvector& x, hvector& v, real coef)
{
  for (size_t i = 0; i < natoms; ++i)
  {
    x[i] += coef * v[i];
    //don't care about pbc
  }
};

// forces computations splitted by the type
void calcDpdForces()
{
  /*
  real dtinvsqrt = 1.0 / sqrt(dtime);
  for (size_t i = 0; i < natoms; ++i)
  {
    for (size_t j = i + 1; j < natoms; ++j)
    {
      real del[] = {xp[i] - xp[j], yp[i] - yp[j], zp[i] - zp[j]};
      minImage(del);

      real rsq = norm2(del);
      if (rsq < cutsq)
      {
        real r = sqrt(rsq);
        real rinv = 1.0 / r;
        real delv[] = {xv[i] - xv[j], yv[i] - yv[j], zv[i] - zv[j]};

        real dot = innerProd(del, delv);
        real randnum = dgauss(gen);

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

        xa[j] -= del[0] * fpair;
        ya[j] -= del[1] * fpair;
        za[j] -= del[2] * fpair;
      }
    }
  }*/
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

void computeForces()
{
  thrust::fill(xa.begin(), xa.end(), 0.0);
  thrust::fill(ya.begin(), ya.end(), 0.0);
  thrust::fill(za.begin(), za.end(), 0.0);

  calcDpdForces();
  calcBondForcesWLC();
  calcAngleForcesBend();

  //addStretchForce();
  addDrivingForce();
}

// ******* CUDA code *********
// NumBlocks is Nrings, NThreadsPerblock == NparticlesPerRing

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

__global__ void kernelBonds(float* d_xp, float* d_yp, float* d_zp, float* d_xa, float* d_ya, float* d_za, int natoms)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  size_t indRing = blockIdx.x;
  size_t indLocal = threadIdx.x;

  if (tid > natoms) return;

  size_t i1 = natomsPerRing * indRing + indLocal;
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

  // finally modify forces
  d_xa[i1] += del[0] * fbond;
  d_ya[i1] += del[1] * fbond;
  d_za[i1] += del[2] * fbond;

  d_xa[i2] -= del[0] * fbond;
  d_ya[i2] -= del[1] * fbond;
  d_za[i2] -= del[2] * fbond;
}

void cuda_calcBondForcesWLC(dvector& d_xp, dvector& d_yp, dvector& d_zp, dvector& d_xa, dvector& d_ya, dvector& d_za)
{
  int nThreadsPerBlock = natomsPerRing;
  int nBlocks = nrings;

  kernelBonds<<< nBlocks, nThreadsPerBlock >>>(
      thrust::raw_pointer_cast(&d_xp[0]), thrust::raw_pointer_cast(&d_yp[0]), thrust::raw_pointer_cast(&d_zp[0]),
      thrust::raw_pointer_cast(&d_xa[0]), thrust::raw_pointer_cast(&d_ya[0]), thrust::raw_pointer_cast(&d_za[0]),
      nrings * natomsPerRing);
}

__global__ void kernelAngles(float* d_xp, float* d_yp, float* d_zp, float* d_xa, float* d_ya, float* d_za, int natoms)
{
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

  float f1[] = {a11 * del1[0] + a12 * del2[0], a11 * del1[1] + a12 * del2[1], a11 * del1[2] + a12 * del2[2]};
  float f3[] = {a22 * del2[0] + a12 * del1[0], a22 * del2[1] + a12 * del1[1], a22 * del2[2] + a12 * del1[2]};

  // apply force to each of 3 atoms
  d_xa[i1] += f1[0];
  d_ya[i1] += f1[1];
  d_za[i1] += f1[2];

  d_xa[i2] -= f1[0] + f3[0];
  d_ya[i2] -= f1[1] + f3[1];
  d_za[i2] -= f1[2] + f3[2];

  d_xa[i3] += f3[0];
  d_ya[i3] += f3[1];
  d_za[i3] += f3[2];
}

void cuda_calcAngleForcesBend(dvector& d_xp, dvector& d_yp, dvector& d_zp, dvector& d_xa, dvector& d_ya, dvector& d_za)
{
  int nThreadsPerBlock = natomsPerRing;
  int nBlocks = nrings;

  kernelAngles<<< nBlocks, nThreadsPerBlock >>>(
      thrust::raw_pointer_cast(&d_xp[0]), thrust::raw_pointer_cast(&d_yp[0]), thrust::raw_pointer_cast(&d_zp[0]),
      thrust::raw_pointer_cast(&d_xa[0]), thrust::raw_pointer_cast(&d_ya[0]), thrust::raw_pointer_cast(&d_za[0]),
      nrings * natomsPerRing);
}

void cuda_computeForces()
{
  // for consistency use h_ prefix for host, d_ for device in this routine
  dvector d_xa(natoms), d_ya(natoms), d_za(natoms);

  thrust::fill(d_xa.begin(), d_xa.end(), 0.0);
  thrust::fill(d_ya.begin(), d_ya.end(), 0.0);
  thrust::fill(d_za.begin(), d_za.end(), 0.0);

  dvector d_xp(xp), d_yp(yp), d_zp(zp);

  //cuda_calcDpdForces();
  cuda_calcBondForcesWLC(d_xp, d_yp, d_zp, d_xa, d_ya, d_za);
  cuda_calcAngleForcesBend(d_xp, d_yp, d_zp, d_xa, d_ya, d_za);

  thrust::fill(xa.begin(), xa.end(), 0.0);
  thrust::fill(ya.begin(), ya.end(), 0.0);
  thrust::fill(za.begin(), za.end(), 0.0);
  calcBondForcesWLC();
  calcAngleForcesBend();

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

// ****************************

void pbcPerAtomsPerDim(size_t ind, hvector& coord)
{
  real boxlo = -0.5 * boxLength;
  real boxhi = 0.5 * boxLength;
  if (coord[ind] < boxlo) {
    coord[ind] += boxLength;
  }
  if (coord[ind] >= boxhi) {
    coord[ind] -= boxLength;
    coord[ind] = std::max(coord[ind], boxlo);
  }
}

void pbc()
{
  for (size_t i = 0; i < natoms; ++i) {
    pbcPerAtomsPerDim(i, xp);
    pbcPerAtomsPerDim(i, yp);
    pbcPerAtomsPerDim(i, zp);
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

  for (size_t timeStep = 0; timeStep < timeEnd; ++timeStep)
  {
    if (timeStep % outEvery == 0)
    {
      std::cout << "t=" << timeStep << std::endl;
      //computeDiams();
      //printStatistics(fstat, timeStep);
    }

    up(xv, xa, dtime * 0.5);
    up(yv, ya, dtime * 0.5);
    up(zv, za, dtime * 0.5);

    up_enforce(xp, xv, dtime);
    up_enforce(yp, yv, dtime);
    up_enforce(zp, zv, dtime);

    pbc();
    if (timeStep % outEvery == 0)
      lammps_dump("evolution.dump", &xp.front(), &yp.front(), &zp.front(), natoms, timeStep, boxLength);

    cuda_computeForces();

    up(xv, xa, dtime * 0.5);
    up(yv, ya, dtime * 0.5);
    up(zv, za, dtime * 0.5);
  }

  fclose(fstat);
  std::cout << "Ended computing" << std::endl;
  return 0;
}


