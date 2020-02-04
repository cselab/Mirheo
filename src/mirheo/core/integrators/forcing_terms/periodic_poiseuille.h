#pragma once

#include <mirheo/core/datatypes.h>
#include <mirheo/core/pvs/particle_vector.h>

#include <mirheo/core/utils/cpu_gpu_defines.h>
#include <mirheo/core/utils/helper_math.h>
#include <mirheo/core/utils/reflection.h>

namespace mirheo
{

/**
 * Apply equal but opposite forces in two halves of the global domain.
 */
class Forcing_PeriodicPoiseuille
{
public:
    enum class Direction {x, y, z};

    /**
     * Setup the forcing term
     *
     * @param magnitude force magnitude to be applied
     * @param dir force will be applied parallel to the specified axis.
     */
    Forcing_PeriodicPoiseuille(real magnitude, Direction dir) :
        magnitude_(magnitude),
        dir_(dir)
    {}

    void setup(ParticleVector* pv, __UNUSED real t)
    {
        domain_ = pv->getState()->domain;
    }

    /**
     * If the force is parallel to \e x axis, its sign will depend on \e y
     * coordinate of a particle (\f$ \vec D \f$ is the global domain size,
     * ParticleVector::Domain::globalSize):
     *
     * \f$ f_x = \begin{cases}
     *  M, & y_p > D_y / 2 \\
     *  -M, & y_p \leqslant D_y / 2
     * \end{cases}
     * \f$
     *
     * Similarly, if the force is parallel to \e y axis, its sign will
     * depend on \e z, parallel to \e z -- will depend on \e x
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
    real magnitude_;
    Direction dir_;

    DomainInfo domain_;

    friend MemberVars<Forcing_PeriodicPoiseuille>;
};

MIRHEO_MEMBER_VARS_2(Forcing_PeriodicPoiseuille, magnitude_, dir_);

} // namespace mirheo
