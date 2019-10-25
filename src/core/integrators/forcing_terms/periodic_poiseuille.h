#pragma once

#include <core/datatypes.h>
#include <core/pvs/particle_vector.h>

#include <core/utils/cpu_gpu_defines.h>
#include <core/utils/helper_math.h>

class ParticleVector;

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
        magnitude(magnitude)
    {
        switch (dir)
        {
            case Direction::x: _dir = 0; break;
            case Direction::y: _dir = 1; break;
            case Direction::z: _dir = 2; break;
        }
    }

    void setup(ParticleVector* pv, __UNUSED real t)
    {
        domain = pv->state->domain;
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
        real3 gr = domain.local2global(p.r);
        real3 ef{0.0f,0.0f,0.0f};

        if (_dir == 0) ef.x = gr.y > 0.5f*domain.globalSize.y ? magnitude : -magnitude;
        if (_dir == 1) ef.y = gr.z > 0.5f*domain.globalSize.z ? magnitude : -magnitude;
        if (_dir == 2) ef.z = gr.x > 0.5f*domain.globalSize.x ? magnitude : -magnitude;

        return ef + original;
    }

private:
    real magnitude;
    int _dir;

    DomainInfo domain;
};
