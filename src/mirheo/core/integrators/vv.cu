#include "vv.h"
#include "integration_kernel.h"

#include "forcing_terms/none.h"
#include "forcing_terms/const_dp.h"
#include "forcing_terms/periodic_poiseuille.h"

#include <mirheo/core/logger.h>
#include <mirheo/core/pvs/particle_vector.h>
#include <mirheo/core/utils/config.h>
#include <mirheo/core/utils/reflection.h>

namespace mirheo
{

template<class ForcingTerm>
IntegratorVV<ForcingTerm>::IntegratorVV(const MirState *state, const std::string& name, ForcingTerm forcingTerm) :
    Integrator(state, name),
    forcingTerm_(forcingTerm)
{}

template<class ForcingTerm>
IntegratorVV<ForcingTerm>::~IntegratorVV() = default;

template<class ForcingTerm>
void IntegratorVV<ForcingTerm>::saveSnapshotAndRegister(Saver& saver)
{
    std::string typeName = constructTypeName<ForcingTerm>("IntegratorVV");
    saver.registerObject<IntegratorVV<ForcingTerm>>(
            this, _saveSnapshot(saver, typeName));
}

template<class ForcingTerm>
ConfigObject IntegratorVV<ForcingTerm>::_saveSnapshot(
        Saver& saver, const std::string& typeName)
{
    ConfigObject config = Integrator::_saveSnapshot(saver, typeName);
    config.emplace("forcingTerm", saver(forcingTerm_));
    return config;
}

template<class ForcingTerm>
void IntegratorVV<ForcingTerm>::stage1(__UNUSED ParticleVector *pv, __UNUSED cudaStream_t stream)
{}

/**
 * The new coordinates and velocities of a particle will be computed
 * as follows:
 * \f$
 * \begin{cases}
 *  f'_p = ForcingTerm(f_p, x_p, v_p) \\
 *  v_{new} = v_p + \dfrac{f'_p}{m_p}  \delta t \\
 *  x_{new} = x_p + v_{new} \, \delta t
 * \end{cases}
 * \f$
 *
 * @tparam ForcingTerm is a functor that can modify computed force
 * per particles (typically add some force field). It has to
 * provide two functions:
 * - This function will be called once before integration and
 *   allows the functor to obtain required variables or data
 *   channels from the ParticleVector:
 *   \code setup(ParticleVector* pv, real t) \endcode
 *
 * - This should be a \c \_\_device\_\_ operator that modifies
 *   the force. It will be called for each particle during the
 *   integration:
 *   \code real3 operator()(real3 f0, Particle p) const \endcode
 *
 */
template<class ForcingTerm>
void IntegratorVV<ForcingTerm>::stage2(ParticleVector *pv, cudaStream_t stream)
{
    const auto t  = static_cast<real>(getState()->currentTime);
    const auto dt = static_cast<real>(getState()->dt);
    
    static_assert(std::is_same<decltype(forcingTerm_.setup(pv, t)), void>::value,
            "Forcing term functor must provide member"
            "void setup(ParticleVector*, real)");

    auto& fterm = forcingTerm_;
    fterm.setup(pv, t);

    auto st2 = [fterm] __device__ (Particle& p, real3 f, real invm, real dt)
    {
        const real3 modF = fterm(f, p);

        p.u += modF * invm * dt;
        p.r += p.u * dt;
    };

    integrate(pv, dt, st2, stream);
    invalidatePV_(pv);
}

template class IntegratorVV<Forcing_None>;
template class IntegratorVV<Forcing_ConstDP>;
template class IntegratorVV<Forcing_PeriodicPoiseuille>;

} // namespace mirheo
