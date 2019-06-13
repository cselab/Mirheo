#pragma once

#include "interface.h"

#include <core/containers.h>

class RodVector;

/**
 * Implements bounce-back from deformable rod.
 */
class BounceFromRod : public Bouncer
{
public:

    BounceFromRod(const YmrState *state, std::string name, float radius, float kbT);
    ~BounceFromRod();

    std::vector<std::string> getChannelsToBeExchanged() const override;
    
private:
    template <typename T>
    struct CollisionTableWrapper
    {
        PinnedBuffer<int> nCollisions{1};
        DeviceBuffer<T> collisionTable;
    };

    /**
     * Maximum supported number of collisions per step
     * will be #bouncesPerSeg * total number of triangles in mesh
     */
    const float coarseCollisionsPerSeg = 5.0f;
    const float fineCollisionsPerSeg = 1.0f;

    CollisionTableWrapper<int2> coarseTable, fineTable;

    // times stored as int so that we can use atomicMax
    // note that times are always positive, thus guarantees ordering
    DeviceBuffer<int> collisionTimes;

    float kbT;
    float radius;

    RodVector *rv;

    void exec(ParticleVector *pv, CellList *cl, bool local, cudaStream_t stream) override;
    void setup(ObjectVector *ov) override;
};
