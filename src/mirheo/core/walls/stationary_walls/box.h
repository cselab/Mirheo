#pragma once

#include <mirheo/core/domain.h>
#include <mirheo/core/datatypes.h>

#include <mirheo/core/utils/cpu_gpu_defines.h>
#include <mirheo/core/utils/helper_math.h>

namespace mirheo
{

class ParticleVector;

/// \brief Represents a box shape.
class StationaryWallBox
{
public:
    /** \brief Construct a StationaryWallBox.
        \param [in] lo Lower bounds of the box (in global coordinates).
        \param [in] hi Upper bounds of the box (in global coordinates).
        \param [in] inside Domain is inside the box if set to \c true.
     */
    StationaryWallBox(real3 lo, real3 hi, bool inside) :
        lo_(lo),
        hi_(hi),
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
    const StationaryWallBox& handler() const
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

        const real3 dist3 = math::min(math::abs(gr - lo_), math::abs(hi_ - gr));
        const real dist = math::min(dist3.x, math::min(dist3.y, dist3.z));

        real sign = 1.0_r;
        if (lo_.x < gr.x && gr.x < hi_.x  &&
            lo_.y < gr.y && gr.y < hi_.y  &&
            lo_.z < gr.z && gr.z < hi_.z)
            sign = -1.0_r;

        return inside_ ? sign * dist : -sign * dist;
    }

private:
    real3 lo_;
    real3 hi_;
    bool inside_;
    DomainInfo domain_;
};

} // namespace mirheo
