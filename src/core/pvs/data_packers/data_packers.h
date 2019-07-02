#pragma once

#include "generic_packer.h"

#include <core/pvs/particle_vector.h>
#include <core/pvs/object_vector.h>
#include <core/pvs/rod_vector.h>

struct ParticlePackerHandler
{
    GenericPackerHandler particles;
};

struct ObjectPackerHandler : public ParticlePackerHandler
{
    GenericPackerHandler objects;
};

struct RodPackerHandler : public ObjectPackerHandler
{
    GenericPackerHandler bisegments;
};



class ParticlePacker
{
public:
    void update(LocalParticleVector *lpv, PackPredicate& predicate, cudaStream_t stream);

    ParticlePackerHandler handler();
    
protected:
    GenericPacker particleData;
};

class ObjectPacker : public ParticlePacker
{
public:
    void update(LocalObjectVector *lov, PackPredicate& predicate, cudaStream_t stream);

    ObjectPackerHandler handler();

protected:
    GenericPacker objectData;
};

class RodPacker : public ObjectPacker
{
public:
    void update(LocalRodVector *lrv, PackPredicate& predicate, cudaStream_t stream);

    RodPackerHandler handler();

protected:
    GenericPacker bisegmentData;
};
