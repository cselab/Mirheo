#pragma once

#include "interface.h"

namespace mirheo
{

/** \brief Advance individual particles with Velocity-Verlet scheme.
    
    \tparam ForcingTerm a functor that can add additional force to the particles 
            depending on their position
 */
template<class ForcingTerm>
class IntegratorVV : public Integrator
{
public:
    /** \param [in] state The global state of the system. The time step and domain used during the execution are passed through this object.
        \param [in] name The name of the integrator.
        \param [in] forcingTerm Additional force added to the particles.
    */
    IntegratorVV(const MirState *state, const std::string& name, ForcingTerm forcingTerm);
    ~IntegratorVV();
    void saveSnapshotAndRegister(Saver&);

    void execute(ParticleVector *pv, cudaStream_t stream) override;

protected:
    ConfigObject _saveSnapshot(Saver&, const std::string& typeName);

private:
    ForcingTerm forcingTerm_;
};

} // namespace mirheo
