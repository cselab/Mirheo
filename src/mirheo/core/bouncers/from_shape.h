// Copyright 2020 ETH Zurich. All Rights Reserved.
#pragma once

#include "interface.h"
#include "kernels/api.h"

#include <random>

namespace mirheo
{

/** \brief Bounce particles against an RigidShapedObjectVector.
    \tparam Shape A class following the AnalyticShape interface

    Particles are bounced against an analytical shape on each object of the
    attached ObjectVector.
    When bounced, the particles will transfer (atomically) their change of momentum into the force and
    torque of the  rigid objects.

    This class only works with RigidShapedObjectVector<Shape> objects.
    It will fail at setup time if the attached object is not rigid.
 */
template <class Shape>
class BounceFromRigidShape : public Bouncer
{
public:

    /** \brief Construct a BounceFromRigidShape object
        \param [in] state Simulation state
        \param [in] name Name of the bouncer
        \param [in] varBounceKernel How are the particles bounced
        \param [in] verbosity 0: no print; 1 print to console the rescue failures; 2 print to console all failures.
    */
    BounceFromRigidShape(const MirState *state, const std::string& name, VarBounceKernel varBounceKernel, int verbosity);
    ~BounceFromRigidShape();

    /**
       Will ask \p ov to keep its old motions information persistently.
       This method will die if \p ov is not of type RigidObjectVector.
     */
    void setup(ObjectVector *ov) override;


    /**
       Will ask \p pv to keep its old positions (not in persistent mode)
     */
    void setPrerequisites(ParticleVector *pv) override;
    std::vector<std::string> getChannelsToBeExchanged() const override;
    std::vector<std::string> getChannelsToBeSentBack() const override;

protected:

    void exec(ParticleVector *pv, CellList *cl, ParticleVectorLocality locality, cudaStream_t stream) override;

    VarBounceKernel varBounceKernel_; ///< The kernel used to reflect the particles on the surface
    std::mt19937 rng_ {42L}; ///< rng used to update varBounceKernel_

    int verbosity_;
};

} // namespace mirheo
