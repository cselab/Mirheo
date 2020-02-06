#pragma once

#include "interface.h"
#include "kernels/api.h"

#include <random>

namespace mirheo
{

/** \brief Bounce particles against an AnalyticalShape object.
    \tparam Shape AnalyticalShape class

    Particles are bounced against an analytical shape on each object of the
    attached \c ObjectVector.
    When bounced, the particles will transfer (atomically) their change of momentum into the force and 
    torque of the  rigid objects.

    This class only works with \c RigidObjectVector objects.
    It will fail at setup time if the attached object is not rigid.
 */
template <class Shape>
class BounceFromRigidShape : public Bouncer
{
public:

    /** \brief Construct a \c BounceFromRigidShape object
        \param [in] state Simulation state
        \param [in] name Name of the bouncer
        \param [in] varBounceKernel How are the particles bounced
    */
    BounceFromRigidShape(const MirState *state, const std::string& name, VarBounceKernel varBounceKernel);
    ~BounceFromRigidShape();

    void setup(ObjectVector *ov) override;

    void setPrerequisites(ParticleVector *pv) override;
    std::vector<std::string> getChannelsToBeExchanged() const override;
    std::vector<std::string> getChannelsToBeSentBack() const override;
    
protected:

    void exec(ParticleVector *pv, CellList *cl, ParticleVectorLocality locality, cudaStream_t stream) override;

    VarBounceKernel varBounceKernel_; ///< The kernel used to reflect the particles on the surface
    std::mt19937 rng_ {42L}; ///< rng used to update \ref varBounceKernel_
};

} // namespace mirheo
