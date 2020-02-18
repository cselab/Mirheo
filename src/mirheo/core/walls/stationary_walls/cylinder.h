#pragma once

#include <mirheo/core/domain.h>
#include <mirheo/core/datatypes.h>

#include <mirheo/core/utils/cpu_gpu_defines.h>
#include <mirheo/core/utils/helper_math.h>

namespace mirheo
{

class ParticleVector;

/// \brief Represents a cylinder along one of the main axes.
class StationaryWallCylinder
{
public:
    /// Represents the direction of the main axis of the cylinder.
    enum class Direction {x, y, z};

    /** \brief Construct a \c StationaryWallCylinder.
        \param [in] center Center of the cylinder in global coordinates in the plane 
                           perpendicular to the direction 
        \param [in] radius Radius of the cylinder
        \param [in] dir The direction of the main axis.
        \param [in] inside Domain is inside the cylinder if set to true.
     */
    StationaryWallCylinder(real2 center, real radius, Direction dir, bool inside) :
        center_(center),
        radius_(radius),
        dir_(dir),
        inside_(inside)
    {}

    /** \brief Synchronize internal state with simulation
        \param [in] comm MPI carthesia communicator
        \param [in] domain Domain info
    */
    void setup(__UNUSED MPI_Comm& comm, DomainInfo domain)
    {
        domain_ = domain;
    }

    /// Get a handler of the shape representation usable on the device
    const StationaryWallCylinder& handler() const
    {
        return *this;
    }

    /** \brief Get the SDF of the current shape at a given position.
        \param [in] r position in local coordinates
        \return The SDF value
    */
    __D__ inline real operator()(real3 r) const
    {
        const real3 gr = domain_.local2global(r);

        real2 projR;
        if (dir_ == Direction::x) projR = make_real2(gr.y, gr.z);
        if (dir_ == Direction::y) projR = make_real2(gr.x, gr.z);
        if (dir_ == Direction::z) projR = make_real2(gr.x, gr.y);

        const real2 dr = projR - center_;
        const real dist = math::sqrt(dot(dr, dr));

        return inside_ ? dist - radius_ : radius_ - dist;
    }

private:
    real2 center_;
    real radius_;
    Direction dir_;
    bool inside_;

    DomainInfo domain_;
};

} // namespace mirheo
