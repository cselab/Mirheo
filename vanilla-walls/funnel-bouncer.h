/*
 *  funnel-bouncer.h
 *  Part of CTC/vanilla-walls/
 *
 *  Created and authored by Diego Rossinelli on 2014-08-11.
 *  Copyright 2015. All rights reserved.
 *
 *  Users are NOT authorized
 *  to employ the present software for their own publications
 *  before getting a written permission from the author of this file.
 */

#pragma once

#include <vector>
#include <tuple>
#include "particles.h"

using namespace std;

struct Bouncer
{
    Particles frozen; //TODO consider moving to Sandwich

    Bouncer(const float L): frozen(0, L) {}
    virtual ~Bouncer() {}
    virtual void _mark(bool * const freeze, Particles p) const = 0;
    virtual void bounce(Particles& dest, const float dt) const = 0;
    virtual void compute_forces(const float kBT, const double dt, Particles& freeParticles) const = 0;

    virtual Particles carve(const Particles& p) //TODO consider moving to Sandwich the implementation
    {
        bool * const freeze = new bool[p.n];

        _mark(freeze, p);

        Particles partition[2] = {Particles(0, p.L[0]), Particles(0, p.L[0])};

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

struct SandwichBouncer: Bouncer
{
    float half_width = 1;

    SandwichBouncer( const float L);

    bool _handle_collision(float& x, float& y, float& z,
               float& u, float& v, float& w,
               float& dt) const;

    void bounce(Particles& dest, const float _dt) const;

    void _mark(bool * const freeze, Particles p) const;

    void compute_forces(const float kBT, const double dt, Particles& freeParticles) const;
};

// TODO compute index and than sort atoms in frozen layer according to it
// for now use this index for filtering only
class AngleIndex
{
    std::vector<int> index;
    float sectorSz;
    size_t nSectors;

    // give angle between [0, 2PI]
    float getAngle(const float x, const float y) const
    {
        return atan2(y, x) + M_PI;
    }

public:

    AngleIndex(float rc, float y0)
    : sectorSz(0.0f), nSectors(0.0f)
    {
        assert(y0 < 0);
        sectorSz = 2.0f * asin(rc/sqrt(-y0));
        nSectors = static_cast<size_t>(2.0f * M_PI) / sectorSz + 1;
    }

    void run(const std::vector<float>& xp, const std::vector<float>& yp)
    {
        index.resize(xp.size());
        for (size_t i = 0; i < index.size(); ++i)
            index[i] = computeIndex(xp[i], yp[i]);
    }

    bool isClose(int srcAngInd, size_t frParticleInd) const
    {
        int destAngInd = getIndex(frParticleInd);
        return destAngInd == srcAngInd
                || (destAngInd + 1)%nSectors == srcAngInd
                || (destAngInd + nSectors - 1)%nSectors == srcAngInd;
    }

    int computeIndex(const float x, const float y) const
    {
        float angle = getAngle(x, y);
        assert(angle >= 0.0f && angle <= 2.0f * M_PI);
        return trunc(angle/sectorSz);
    }

    int getIndex(size_t i) const { return index[i]; }
};

struct TomatoSandwich: SandwichBouncer
{
    float xc = 0, yc = 0, zc = 0;
    float radius2 = 1;

    const float rc = 1.0;

    RowFunnelObstacle funnelLS;
    Particles frozenLayer[3]; // three layers every one is rc width
    AngleIndex angleIndex[3];

    TomatoSandwich(const float boxLength);

    Particles carve(const Particles& particles);

    void _mark(bool * const freeze, Particles p) const;
 
    float _compute_collision_time(const float _x0, const float _y0,
			  const float u, const float v, 
			  const float xc, const float yc, const float r2);

    bool _handle_collision(float& x, float& y, float& z,
			   float& u, float& v, float& w,
			   float& dt) const;
    
    void bounce(Particles& dest, const float _dt) const;
 void compute_forces(const float kBT, const double dt, Particles& freeParticles) const;

 void vmd_xyz(const char * path) const;

private:
    //dpd forces computations related methods
    Particles carveLayer(const Particles& input, size_t indLayer, float bottom, float top);
    Particles carveAllLayers(const Particles& p);
    void computeDPDPairForLayer(const float kBT, const double dt, int i,
            const float* coord, const float* vel, float* df, const float offsetX,
            int seed1, int frLayerSaruTagOffset) const;
    void computePairDPD(const float kBT, const double dt, Particles& freeParticles) const;
    void dpd_forces_1particle(size_t layerIndex, const float kBT, const double dt,
            int i, const float* offset, const float* coord, const float* vel, float* df,
            const int giddstart, const int frLayerSaruTagOffset) const;
};
