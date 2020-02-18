#pragma once

#include <mirheo/core/datatypes.h>
#include <mirheo/core/pvs/particle_vector.h>
#include <mirheo/core/utils/cpu_gpu_defines.h>
#include <mirheo/core/utils/helper_math.h>
#include <mirheo/core/utils/reflection.h>

namespace mirheo
{

class ParticleVector;

/**\brief Apply equal but opposite forces in two halves of the global domain.

    \rst

    .. math::

        f_x = \begin{cases} 
        F, & y_p > L_y / 2 \\
        -F, & y_p \leqslant L_y / 2
        \end{cases}
    
    Similarly, if the force is parallel to the y axis, its sign will
    depend on z; parallel to z it will depend on x.
    \endrst
*/
class ForcingTermPeriodicPoiseuille
{
public:
    /** \brief Encode directions
     */
    enum class Direction {x, y, z};

    /** \brief Construct a \c ForcingTermPeriodicPoiseuille object
        \param magnitude force magnitude to be applied.
        \param dir The force will be applied parallel to the specified axis.
     */
    ForcingTermPeriodicPoiseuille(real magnitude, Direction dir) :
        magnitude_(magnitude),
        dir_(dir)
    {}

    /**\brief Initialize internal state
       \param [in] pv the \c ParticleVector that will be updated
       \param [in] t Current simulation time

       This method must be called at every time step.
    */
    void setup(ParticleVector* pv, __UNUSED real t)
    {
        domain_ = pv->getState()->domain;
    }

    /**\brief Add the additional force to the current one on a particle
       \param [in] original Original force acting on the particle
       \param [in] p Particle on which to apply the additional force
       \return The total force that must be applied to the particle
    */
    __D__ inline real3 operator()(real3 original, Particle p) const
    {
        const real3 gr = domain_.local2global(p.r);
        real3 ef {0.0_r, 0.0_r, 0.0_r};

        if (dir_ == Direction::x) ef.x = gr.y > 0.5_r * domain_.globalSize.y ? magnitude_ : -magnitude_;
        if (dir_ == Direction::y) ef.y = gr.z > 0.5_r * domain_.globalSize.z ? magnitude_ : -magnitude_;
        if (dir_ == Direction::z) ef.z = gr.x > 0.5_r * domain_.globalSize.x ? magnitude_ : -magnitude_;

        return ef + original;
    }

private:
    real magnitude_; ///< force magnitude
    Direction dir_;  ///< direction of the force

    DomainInfo domain_; ///< domain info

    friend MemberVars<ForcingTermPeriodicPoiseuille>;
};

MIRHEO_MEMBER_VARS(2, ForcingTermPeriodicPoiseuille, magnitude_, dir_);

} // namespace mirheo
