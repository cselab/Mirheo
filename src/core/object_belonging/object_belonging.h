#pragma once

#include "interface.h"
#include <core/containers.h>

enum class BelongingTags
{
    Outside = 0, Inside
};

class ObjectBelongingChecker_Common : ObjectBelongingChecker
{
public:
    std::string name;

    ObjectBelongingChecker_Common(std::string name) : ObjectBelongingChecker(name) { }

    /**
     * Particle with tags == BelongingTags::Outside  will be copied to pvOut
     *                    == BelongingTags::Inside   will be copied to pvIn
     * Other particles are DROPPED (boundary particles)
     */
    void splitByBelonging(ParticleVector* src, ParticleVector* pvIn, ParticleVector* pvOut, cudaStream_t stream) override;
    void checkInner(ParticleVector* pv, CellList* cl, cudaStream_t stream) override;
    void setup(ObjectVector* ov) override { this->ov = ov; }


protected:
    ObjectVector* ov;

    PinnedBuffer<BelongingTags> tags;
    PinnedBuffer<int> nInside{1}, nOutside{1};

    virtual void tagInner(ParticleVector* pv, CellList* cl, cudaStream_t stream) = 0;
};
