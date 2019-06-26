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

    BounceFromRod(const MirState *state, std::string name, float radius);
    ~BounceFromRod();

    void setPrerequisites(ParticleVector *pv) override;
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
    const float collisionsPerSeg = 5.0f;

    CollisionTableWrapper<int2> table;

    // times stored as int so that we can use atomicMax
    // note that times are always positive, thus guarantees ordering
    DeviceBuffer<int> collisionTimes;

    float radius;

    RodVector *rv;

    void exec(ParticleVector *pv, CellList *cl, bool local, cudaStream_t stream) override;
    void setup(ObjectVector *ov) override;
};
