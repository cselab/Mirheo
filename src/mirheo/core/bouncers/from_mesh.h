// Copyright 2020 ETH Zurich. All Rights Reserved.
#pragma once

#include "interface.h"
#include "kernels/api.h"

#include <mirheo/core/containers.h>

#include <random>

namespace mirheo
{

class RigidObjectVector;


/** \brief Bounce particles against a triangle mesh.

    - if the attached object is a RigidObjectVector, the bounced particles will
      transfer (atomically) their change of momentum into the force and torque of the
      rigid object.

    - if the attached object is a not RigidObjectVector, the bounced particles will
      transfer (atomically) their change of momentum into the force of the three vertices
      which form the colliding triangle.

      This class will fail if the object does not have a mesh representing its surfece.
 */
class BounceFromMesh : public Bouncer
{
public:

    /** \brief Construct a BounceFromMesh object
        \param [in] state Simulation state
        \param [in] name Name of the bouncer
        \param [in] varBounceKernel How are the particles bounced
    */
    BounceFromMesh(const MirState *state, const std::string& name, VarBounceKernel varBounceKernel);
    ~BounceFromMesh();

    /**
        If \p ov is a rigid object, this will ask it to keep its old motions accross exchangers.
        Otherwise, ask \p ov to keep its old positions accross exchangers.
    */
    void setup(ObjectVector *ov) override;

    /**
       Will ask \p pv to keep its old positions (not in persistent mode)
     */
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

    const real coarseCollisionsPerTri_ = 5.0_r; ///< maximum average possible number of collision per triangle in one step
    const real fineCollisionsPerTri_   = 1.0_r; ///< maximum average number of collision per triangle in one step

    CollisionTableWrapper<int2> coarseTable_; ///< collision table for the first step
    CollisionTableWrapper<int2> fineTable_;   ///< collision table for the second step

    /** times stored as int so that we can use atomicMax
        note that times are always positive, thus guarantees ordering
    */
    DeviceBuffer<int> collisionTimes_;

    VarBounceKernel varBounceKernel_;  ///< The kernel used to reflect the particles on the surface
    std::mt19937 rng_ {42L}; ///< rng used to update varBounceKernel_

    RigidObjectVector *rov_;

    void exec(ParticleVector *pv, CellList *cl, ParticleVectorLocality locality, cudaStream_t stream) override;
};

} // namespace mirheo
