#pragma once

#include "interface.h"

#include <core/containers.h>

class RigidObjectVector;


/**
 * Implements bounce-back from deformable mesh.
 * Mesh vertices must be the particles in the ParicleVector
 */
class BounceFromMesh : public Bouncer
{
public:

    BounceFromMesh(const MirState *state, std::string name, float kbT);
    ~BounceFromMesh();

    void setPrerequisites(ParticleVector *pv) override;
    std::vector<std::string> getChannelsToBeExchanged() const override;
    
private:
    template<typename T>
    struct CollisionTableWrapper
    {
        PinnedBuffer<int> nCollisions{1};
        DeviceBuffer<T> collisionTable;
    };

    /**
     * Maximum supported number of collisions per step
     * will be #bouncesPerTri * total number of triangles in mesh
     */
    const float coarseCollisionsPerTri = 5.0f;
    const float fineCollisionsPerTri = 1.0f;

    CollisionTableWrapper<int2> coarseTable, fineTable;

    // times stored as int so that we can use atomicMax
    // note that times are always positive, thus guarantees ordering
    DeviceBuffer<int> collisionTimes;

    float kbT;

    RigidObjectVector *rov;

    void exec(ParticleVector *pv, CellList *cl, bool local, cudaStream_t stream) override;
    void setup(ObjectVector *ov) override;
};
