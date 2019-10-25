#pragma once

#include "interface.h"
#include "kernels/api.h"

#include <core/containers.h>

#include <random>

class RodVector;

/**
 * Implements bounce-back from deformable rod.
 */
class BounceFromRod : public Bouncer
{
public:

    BounceFromRod(const MirState *state, const std::string& name, real radius, VarBounceKernel varBounceKernel);
    ~BounceFromRod();

    void setPrerequisites(ParticleVector *pv) override;
    std::vector<std::string> getChannelsToBeExchanged() const override;
    std::vector<std::string> getChannelsToBeSentBack() const override;
    
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
    const real collisionsPerSeg = 5.0f;

    CollisionTableWrapper<int2> table;

    // times stored as int so that we can use atomicMax
    // note that times are always positive, thus guarantees ordering
    DeviceBuffer<int> collisionTimes;

    real radius;

    RodVector *rv;

    VarBounceKernel varBounceKernel;
    std::mt19937 rng {42L};
    
    void exec(ParticleVector *pv, CellList *cl, ParticleVectorLocality locality, cudaStream_t stream) override;
    void setup(ObjectVector *ov) override;
};
