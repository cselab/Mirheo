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

using namespace std;

typedef double real;
typedef size_t sizeType;

typedef std::vector<real> hvector;

// ************ global variables *****************
const real boxLength = 10.0;

const sizeType nrings = 0;
const sizeType natomsPerRing = 10;
const sizeType nFreeFluidAtoms = 2700;//for sphere 2900; //for tube 590;
const sizeType nFrozenFluidAtoms = 300;//for sphere 100; //for tube 564;
const sizeType nfluidAtoms = nFreeFluidAtoms + nFrozenFluidAtoms;

const sizeType nFreeParticles = nrings * natomsPerRing + nFreeFluidAtoms; // only free particles are integrated
const sizeType natoms = nrings * natomsPerRing + nfluidAtoms;

hvector xp(natoms), yp(natoms), zp(natoms),
       xv(natoms), yv(natoms), zv(natoms),
       xa(natoms), ya(natoms), za(natoms);

// 0 for rings
// 1 for free fluid
// 2 for frozen atoms
// For the sake of force calculation first [0, nrings * natomsPerRing) particles
// are for rings, the next N is for free fluid, and the last part - for frozen atoms
// since they should not participate in the integration
std::vector<sizeType> type(natoms);

// dpd parameters
const real dtime = 0.001;
const real kbT = 0.1;
const sizeType timeEnd = 100;

const real a0 = 500.0, gamma0 = 4.5, cut = 1.2, cutsq = cut * cut, kPower = 0.25,
    sigma = sqrt(2.0 * kbT * gamma0);

// WLC bond parameters (assumed DPD length unit is 0.5*real)
const real lambda = 2.5e-4;
const real lmax  = 1.3;

// bending angle parameters
const real kbend = 50.0 * kbT;
const real theta = M_PI - 2.0 * M_PI / natomsPerRing;

// misc parameters
const size_t outEvery = 10;
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
real saru(unsigned int seed1, unsigned int seed2, unsigned int seed3)
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

real h_getGRand(size_t i, size_t j, size_t idtimestep)
{
  const real mysaru = saru(std::min(i, j), std::max(i, j), idtimestep);
  return 3.464101615 * mysaru - 1.732050807;
}
#endif

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
    fprintf(f, "%lu %lu %g %g %g\n", i, type[i], xs[i], ys[i], zs[i]);
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
  /*__host__ __device__*/ real operator()(const real& x, const real& y) const
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

/**
 * All-together code just to show the concept
 */
class LevelSetBB
{
  enum GeomType
  {
    cylinder,
    outSphere,
    betweenPlanes,
    planesAndSphere
  };

  enum IntersectStatus
  {
    inside,
    onsurface,
    outside
  };

  struct Box
  {
    const real low[3];
    const real top[3];
    Box()
    : low{-boxLength/2.0, -boxLength/2.0, -boxLength/2.0},
      top{boxLength/2.0, boxLength/2.0, boxLength/2.0}
    {}
  } m_box;

  const real localTolerance = 0;

  GeomType m_geomType;
  hvector m_lsValuesNew, m_lsValuesOld;
  hvector m_xpOld, m_ypOld, m_zpOld; // buffer for the old positions

public:
  LevelSetBB() : m_geomType(planesAndSphere)
  {
  }

  static real lsCylinder(real xp, real yp, real zp, real radius)
  {
    return (sqrt(xp * xp + yp * yp) - radius);
  }

  static bool isInsideCylinder(real xp, real yp, real zp, real radius)
  {
    return lsCylinder(xp, yp, zp, radius) < 0;
  }

  static real lsOutsideSphere(real xp, real yp, real zp, real radius)
  {
    return (radius - sqrt(xp * xp + yp * yp + zp * zp));
  }

  static bool isOutsideSphere(real xp, real yp, real zp, real radius)
  {
    return lsOutsideSphere(xp, yp, zp, radius) < 0;
  }

  // two planes: p1=(x1,0,0), n1=(1,0,0); p2=(x2,0,0), n1=(-1,0,0)
  // ls for a plane distToPlane = (point - pointOnPlane) * normal;
  static real lsBetwenPlanes(real xp, real yp, real zp, real x1, real x2)
  {
    real d1 = (xp - x1);
    real d2 = -1.0 * (xp - x2);
    return -std::min(d1, d2);
  }

  static bool isBetwenPlanes(real xp, real yp, real zp, real x1, real x2)
  {
    return lsBetwenPlanes(xp, yp, zp, x1, x2) < 0;
  }

  // two planes: p1=(x1,0,0), n1=(1,0,0); p2=(x2,0,0), n1=(-1,0,0)
  // ls for a plane distToPlane = (point - pointOnPlane) * normal;
  static real lsBetwenPlanesAndSphere(real xp, real yp, real zp)
  {
    real radius = 2.0;
    real x1 = -4.5;
    real x2 = 4.5;
    real d1 = lsBetwenPlanes(xp, yp, zp, x1, x2);
    real d2 = lsOutsideSphere(xp, yp, zp, radius);
    return std::max(d1, d2);
  }

  static bool isBetwenPlanesAndSphere(real xp, real yp, real zp)
  {
    return lsBetwenPlanesAndSphere(xp, yp, zp) < 0;
  }

  void run(hvector& xp, hvector& yp, hvector& zp, hvector& xv, hvector& yv, hvector& zv)
  {
    if (m_lsValuesOld.size() == 0) //called first time
    {
      m_lsValuesNew.resize(xp.size(), 0.0);
      precomputeLSvalues(xp, yp, zp);
      saveOldValues(xp, yp, zp);
      assert(m_lsValuesOld.size() != 0);
      return;
    }

    precomputeLSvalues(xp, yp, zp);
    for (size_t i = 0; i < natoms; i++) {
      real distNew = m_lsValuesNew[i];

      //if it is from out of the LS domain if the LS domain is smaller than the whole thing
      if (std::isinf(distNew))
        continue;
      if (checkPoint(distNew) == outside) {
        //turn velocity backward
        xv[i] *= -1.0; yv[i] *= -1.0; zv[i] *= -1.0;

        real distOld = m_lsValuesOld[i];
        if (std::isinf(distOld)) //if it is from out of the LS domain
          continue;

        IntersectStatus prevIntSt = checkPoint(distOld);
        if (prevIntSt == inside || prevIntSt == onsurface) {
          distOld = fabs(distOld);
          distNew = fabs(distNew);
          // x_refl = x_old + scale * (x_new - x_old)
          real scale = (distOld - distNew)/(distOld + distNew);
          real pOld[3];
          getOldPosition(i, pOld);
          real xdif[] = {xp[i] - pOld[0], yp[i] - pOld[1], zp[i] - pOld[2]};

          xdif[0] *= scale; xdif[1] *= scale; xdif[2] *= scale;

          xp[i] = pOld[0] + xdif[0];
          yp[i] = pOld[1] + xdif[1];
          zp[i] = pOld[2] + xdif[2];
        }
      }
    }
    saveOldValues(xp, yp, zp);

    /* check that all free particles are inside
    for (size_t i = 0; i < nFreeParticles; i++) {
      if (!isInsideCylinder(xp[i], yp[i], zp[i], 2.5))
      {
        std::cout << i << ", " << xp[i] << ", " << yp[i] << ", " << sqrt(xp[i]*xp[i] + yp[i]*yp[i]) - 2.5 << std::endl;
      }
    }

    for (size_t i = 0; i < nFreeParticles; i++) {
      if (!isOutsideSphere(xp[i], yp[i], zp[i], 2.0))
      {
        std::cout << i << ", " << xp[i] << ", " << yp[i] << ", " << sqrt(xp[i]*xp[i] + yp[i]*yp[i] + zp[i]*zp[i]) - 2.0 << std::endl;
      }
    }*/

    real radius = 2.0;
    real x1 = -4.5;
    real x2 = 4.5;
    for (size_t i = 0; i < nFreeParticles; i++) {
      bool d1 = isBetwenPlanes(xp[i], yp[i], zp[i], x1, x2);
      bool d2 = isOutsideSphere(xp[i], yp[i], zp[i], radius);
      if (!d1 || !d2))
      {
        std::cout << i << ", " << xp[i] << ", " << yp[i] << std::endl;
      }
    }
  }

private:
  void precomputeLSvalues(const hvector& xp, const hvector& yp, const hvector& zp)
  {
    if (m_geomType == cylinder) {
      real m_radius = 2.5;
      for (size_t i = 0; i < natoms; ++i) {
        if (isInsideBB(xp[i], yp[i], zp[i]))
          m_lsValuesNew[i] = lsCylinder(xp[i], yp[i], zp[i], m_radius);
        else {
          m_lsValuesNew[i] = -std::numeric_limits<real>::infinity();
        }
      }
    } else if (m_geomType == outSphere) {
      // sign is inverted
      real m_radius = 2.0;
      for (size_t i = 0; i < natoms; ++i) {
        if (isInsideBB(xp[i], yp[i], zp[i]))
          m_lsValuesNew[i] = lsOutsideSphere(xp[i], yp[i], zp[i], m_radius);
        else {
          m_lsValuesNew[i] = std::numeric_limits<real>::infinity();
        }
      }
    } else if (m_geomType == betweenPlanes) {
      for (size_t i = 0; i < natoms; ++i) {
        if (isInsideBB(xp[i], yp[i], zp[i]))
          m_lsValuesNew[i] = lsBetwenPlanes(xp[i], yp[i], zp[i], -4.5, 4.5);
        else {
          m_lsValuesNew[i] = -std::numeric_limits<real>::infinity();
        }
      }
    } else if (m_geomType == planesAndSphere) {
      for (size_t i = 0; i < natoms; ++i) {
        if (isInsideBB(xp[i], yp[i], zp[i]))
          m_lsValuesNew[i] = lsBetwenPlanesAndSphere(xp[i], yp[i], zp[i]);
        else {
          m_lsValuesNew[i] = -std::numeric_limits<real>::infinity();
        }
      }
    } else {
      std::cout << "Only cylinder is supported for now\n";
    }
  }

  IntersectStatus checkPoint(double diff)
  {
    if (diff > localTolerance)
      return outside;
    else if (diff < -localTolerance)
      return inside;

    return onsurface;
  }

  void getOldPosition(size_t index, real* pOld)
  {
    // there are two options - compute or use buffer with saved positions
    //pOld[0] = xp - dtime * xv; pOld[1] = yp - dtime * yv; pOld[2] = zp - dtime * zv;
    // for now do the second way around
    pOld[0] = m_xpOld[index];
    pOld[1] = m_ypOld[index];
    pOld[2] = m_zpOld[index];
  }

  bool isInsideBB(const real x, const real y, const real z) const
  {
    if (x >= m_box.low[0] && x <= m_box.top[0]
     && y >= m_box.low[1] && y <= m_box.top[1]
     && z >= m_box.low[2] && z <= m_box.top[2])
      return true;
    return false;
  }

  void saveOldValues(const hvector& xp, const hvector& yp, const hvector& zp)
  {
    m_lsValuesOld = m_lsValuesNew;
    m_xpOld = xp;
    m_ypOld = yp;
    m_zpOld = zp;
  }
};

// **** initialization *****
void addRing(size_t indRing)
{
  real cmass[3];
  getRandPoint(cmass[0], cmass[1], cmass[2]);

  for (sizeType indLocal = 0; indLocal < natomsPerRing; ++indLocal) {
    sizeType i = natomsPerRing * indRing + indLocal;
    real angle = 2.0 * M_PI / natomsPerRing * i;
    xp[i] = ringRadius * cos(angle) + cmass[0];
    yp[i] = ringRadius * sin(angle) + cmass[1];
    zp[i] = cmass[2];
    type[i] = 0;
  }
}

void initPositionsCube()
{
  for (size_t indRing = 0; indRing < nrings; ++indRing) {
    addRing(indRing);
  }

  for (size_t i = nrings * natomsPerRing; i < natoms; ++i) {
    getRandPoint(xp[i], yp[i], zp[i]);
  }
}

void initPositionsFlowInTube()
{
  real radius = 2.5;

  for (sizeType i = nrings * natomsPerRing; i < nFreeFluidAtoms;)
  {
    getRandPoint(xp[i], yp[i], zp[i]);
    if (LevelSetBB::isInsideCylinder(xp[i], yp[i], zp[i], radius)) {
      type[i] = 1;
      ++i;
    }
  }

  for (sizeType i = nFreeFluidAtoms; i < natoms;)
  {
    getRandPoint(xp[i], yp[i], zp[i]);
    if (!LevelSetBB::isInsideCylinder(xp[i], yp[i], zp[i], radius) &&
        LevelSetBB::isInsideCylinder(xp[i], yp[i], zp[i], radius + 1.0)) {
      type[i] = 2;
      ++i;
    }
  }
}

void initPositionsOutSphere()
{
  real radius = 2.0;

  for (sizeType i = nrings * natomsPerRing; i < nFreeFluidAtoms;)
  {
    getRandPoint(xp[i], yp[i], zp[i]);
    if (LevelSetBB::isOutsideSphere(xp[i], yp[i], zp[i], radius)) {
      type[i] = 1;
      ++i;
    }
  }

  for (sizeType i = nFreeFluidAtoms; i < natoms;)
  {
    getRandPoint(xp[i], yp[i], zp[i]);
    if (!LevelSetBB::isOutsideSphere(xp[i], yp[i], zp[i], radius)) {
      type[i] = 2;
      ++i;
    }
  }
}

void initPositionsBetweenPlanesAndSphere()
{
  //real x1 = -4.5, x2 = 4.5;

  for (sizeType i = nrings * natomsPerRing; i < nFreeFluidAtoms;)
  {
    getRandPoint(xp[i], yp[i], zp[i]);
    if (LevelSetBB::isBetwenPlanesAndSphere(xp[i], yp[i], zp[i])) {
      type[i] = 1;
      ++i;
    }
  }

  for (sizeType i = nFreeFluidAtoms; i < natoms;)
  {
    getRandPoint(xp[i], yp[i], zp[i]);
    if (!LevelSetBB::isBetwenPlanesAndSphere(xp[i], yp[i], zp[i])) {
      type[i] = 2;
      ++i;
    }
  }
}

// forces computations splitted by the type
void calcDpdForces(size_t timeStep)
{
  real dtinvsqrt = 1.0 / sqrt(dtime);
  for (sizeType i = 0; i < natoms; ++i)
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

        assert(!std::isnan(xa[i]) && !std::isnan(ya[i]) && !std::isnan(za[i]));
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
  real drivingForceY = 100.0;
  std::for_each(ya.begin(), ya.end(), [&](real& in) { in += drivingForceY; });
}

void computeForces(size_t timeStep)
{
  std::fill(xa.begin(), xa.end(), 0.0);
  std::fill(ya.begin(), ya.end(), 0.0);
  std::fill(za.begin(), za.end(), 0.0);

  calcDpdForces(timeStep);
  //calcBondForcesWLC();
  //calcAngleForcesBend();

  //addStretchForce();
  //addDrivingForce();
}

// initial integration of velocity-verlet
void initialIntegrate()
{
  std::transform(xa.begin(), xa.begin() + nFreeParticles, xv.begin(), xv.begin(), SaxpyOp(dtime * 0.5));
  std::transform(ya.begin(), ya.begin() + nFreeParticles, yv.begin(), yv.begin(), SaxpyOp(dtime * 0.5));
  std::transform(za.begin(), za.begin() + nFreeParticles, zv.begin(), zv.begin(), SaxpyOp(dtime * 0.5));

  std::transform(xv.begin(), xv.begin() + nFreeParticles, xp.begin(), xp.begin(), SaxpyOp(dtime));
  std::transform(yv.begin(), yv.begin() + nFreeParticles, yp.begin(), yp.begin(), SaxpyOp(dtime));
  std::transform(zv.begin(), zv.begin() + nFreeParticles, zp.begin(), zp.begin(), SaxpyOp(dtime));
}

//final integration of velocity-verlet
void finalIntegrate()
{
  std::transform(xa.begin(), xa.begin() + nFreeParticles, xv.begin(), xv.begin(), SaxpyOp(dtime * 0.5));
  std::transform(ya.begin(), ya.begin() + nFreeParticles, yv.begin(), yv.begin(), SaxpyOp(dtime * 0.5));
  std::transform(za.begin(), za.begin() + nFreeParticles, zv.begin(), zv.begin(), SaxpyOp(dtime * 0.5));
}

struct PbcOp {
  /*__host__ __device__*/ void operator()(real& coord) const
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
  FILE * fstat = fopen("diag.txt", "w");

  initPositionsBetweenPlanesAndSphere();
  std::fill(xv.begin(), xv.end(), 0.0);
  std::fill(yv.begin(), yv.end(), 0.0);
  std::fill(zv.begin(), zv.end(), 0.0);

  LevelSetBB lsbb;
  for (size_t timeStep = 0; timeStep < timeEnd; ++timeStep)
  {
    if (timeStep % outEvery == 0)
    {
      std::cout << "t=" << timeStep << std::endl;
      //computeDiams();
      //printStatistics(fstat, timeStep);
    }

    initialIntegrate();
    pbc();
    lsbb.run(xp, yp, zp, xv, yv, zv);
    if (timeStep % outEvery == 0)
      lammps_dump("evolution.dump", &xp.front(), &yp.front(), &zp.front(), natoms, timeStep, boxLength);

    computeForces(timeStep);

    finalIntegrate();

    // check that no nan produced
    for (size_t i = 0; i < natoms; i++)
      if (std::isnan(xp[i]) || std::isnan(yp[i]) || std::isnan(zp[i]))
        assert(false);
  }

  fclose(fstat);
  std::cout << "Ended computing" << std::endl;
  return 0;
}


