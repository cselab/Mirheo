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

    /** \brief Create a \c ConfigObject describing the integrator state and register it in the saver.
        \param [in,out] saver The \c Saver object. Provides save context and serialization functions.

        Checks that the object type is exactly \c IntegratorVV<ForcingTerm>.
      */
    void saveSnapshotAndRegister(Saver& saver);

    void execute(ParticleVector *pv, cudaStream_t stream) override;

protected:
    /** \brief Implementation of the snapshot saving. Reusable by potential derived classes.
        \param [in,out] saver The \c Saver object. Provides save context and serialization functions.
        \param [in] typeName The name of the type being saved.
      */
    ConfigObject _saveSnapshot(Saver& saver, const std::string& typeName);

private:
    ForcingTerm forcingTerm_;
};

} // namespace mirheo
