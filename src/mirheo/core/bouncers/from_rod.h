#pragma once

#include "interface.h"
#include "kernels/api.h"

#include <mirheo/core/containers.h>

#include <random>

namespace mirheo
{

class RodVector;

/** \brief Bounce particles against rods.

    The particles are reflacted against the set of capsules around each segment forming the rod.
    This class will fail if the attached object is not a \c RodObjectVector
 */
class BounceFromRod : public Bouncer
{
public:

    /** \brief Construct a \c BounceFromRod object
        \param [in] state Simulation state
        \param [in] name Name of the bouncer
        \param [in] radius The radius of the capsules attached to each segment 
        \param [in] varBounceKernel How are the particles bounced
    */
    BounceFromRod(const MirState *state, const std::string& name, real radius, VarBounceKernel varBounceKernel);
    ~BounceFromRod();

    /** 
        Ask \p ov to keep its old motions accross persistently. 
        This method will die if \p ov is not of type \c RodObjectVector.
    */
    void setup(ObjectVector *ov) override;

    /**
       Will ask \p pv to keep its old positions (not in persistent mode)
     */
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
     * will be bouncesPerSeg * total number of triangles in mesh
     */
    const real collisionsPerSeg_ = 5.0_r;

    CollisionTableWrapper<int2> table_;

    /**
       times stored as int so that we can use atomicMax
       note that times are always positive, thus guarantees ordering
    */
    DeviceBuffer<int> collisionTimes;

    real radius_;

    RodVector *rv_;

    VarBounceKernel varBounceKernel_;
    std::mt19937 rng_ {42L};
    
    void exec(ParticleVector *pv, CellList *cl, ParticleVectorLocality locality, cudaStream_t stream) override;
};

} // namespace mirheo
