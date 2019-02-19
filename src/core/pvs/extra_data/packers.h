#pragma once

#include "device_packer.h"

#include <core/pvs/particle_vector.h>
#include <core/pvs/object_vector.h>

/**
 * Class that uses DevicePacker to pack a single particle entity; always pack coordinates and velocities
 */
struct ParticlePacker : public DevicePacker
{
    ParticlePacker(ParticleVector *pv, LocalParticleVector *lpv, PackPredicate predicate, cudaStream_t stream);
};

/**
 * Class that uses DevicePacker to pack a single particle entity; do not pack coordinates and velocities
 */
struct ParticleExtraPacker : public DevicePacker
{
    ParticleExtraPacker(ParticleVector *pv, LocalParticleVector *lpv, PackPredicate predicate, cudaStream_t stream);
};


/**
 * Class that uses DevicePacker to pack extra data per object
 */
struct ObjectExtraPacker : public DevicePacker
{
    ObjectExtraPacker(ObjectVector *ov, LocalObjectVector *lov, PackPredicate predicate, cudaStream_t stream);
};


/**
 * Class that uses both ParticlePacker and ObjectExtraPacker
 * to pack everything. Provides totalPackedSize_byte of an object
 */
struct ObjectPacker
{
    ParticlePacker    part;
    ObjectExtraPacker obj;
    int totalPackedSize_byte = 0;

    ObjectPacker(ObjectVector *ov, LocalObjectVector *lov, PackPredicate predicate, cudaStream_t stream);
};

