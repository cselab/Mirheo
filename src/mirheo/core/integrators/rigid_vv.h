#pragma once

#include "interface.h"

namespace mirheo
{

/** \brief Integrate \c RigidObjectVector objects given torque and force.
    \ingroup Integrators
    
    \rst
    Advance the RigidMotion and the frozen particles of the \c RigidObjectVector objects.
    The particles of each object are given the velocities corresponding to the rigid object motion.
    \endrst
 */
class IntegratorVVRigid : public Integrator
{
public:
    /** \param [in] state The global state of the system. The time step and domain used during the execution are passed through this object.
        \param [in] name The name of the integrator.
    */
    IntegratorVVRigid(const MirState *state, const std::string& name);
    ~IntegratorVVRigid();

    void setPrerequisites(ParticleVector *pv) override;

    void execute(ParticleVector *pv, cudaStream_t stream) override;
};

} // namespace mirheo
