#pragma once

#include "interface.h"
#include "kernels/api.h"

#include <mirheo/core/containers.h>

#include <random>

namespace mirheo
{

class RigidObjectVector;


/**
 * Implements bounce-back from deformable mesh.
 * Mesh vertices must be the particles in the ParicleVector
 */
class BounceFromMesh : public Bouncer
{
public:

    BounceFromMesh(const MirState *state, const std::string& name, VarBounceKernel varBounceKernel);
    ~BounceFromMesh();

    void setPrerequisites(ParticleVector *pv) override;
    std::vector<std::string> getChannelsToBeExchanged() const override;
    std::vector<std::string> getChannelsToBeSentBack() const override;
    
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
    const real coarseCollisionsPerTri = 5.0_r;
    const real fineCollisionsPerTri = 1.0_r;

    CollisionTableWrapper<int2> coarseTable, fineTable;

    // times stored as int so that we can use atomicMax
    // note that times are always positive, thus guarantees ordering
    DeviceBuffer<int> collisionTimes;

    VarBounceKernel varBounceKernel;
    std::mt19937 rng {42L};

    RigidObjectVector *rov;

    void exec(ParticleVector *pv, CellList *cl, ParticleVectorLocality locality, cudaStream_t stream) override;
    void setup(ObjectVector *ov) override;
};

} // namespace mirheo
