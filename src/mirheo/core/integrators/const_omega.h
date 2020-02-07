#pragma once

#include "interface.h"

namespace mirheo
{

/** \brief Rotate \c ParticleVector objects with a constant angular velocity.
    \ingroup Integrators

    The center of rotation is defined in the global coordinate system.
 */
class IntegratorConstOmega : public Integrator
{
public:
    /** \brief Construct a \c IntegratorConstOmega object.
        \param [in] state The global state of the system. The time step and domain used during the execution are passed through this object.
        \param [in] name The name of the integrator.
        \param [in] center The center of rotation, in global coordinates system.
        \param [in] omega The angular velocity of rotation.
    */
    IntegratorConstOmega(const MirState *state, const std::string& name, real3 center, real3 omega);
    ~IntegratorConstOmega();

    void execute(ParticleVector *pv, cudaStream_t stream) override;

private:
    real3 center_, omega_;
};

} // namespace mirheo
