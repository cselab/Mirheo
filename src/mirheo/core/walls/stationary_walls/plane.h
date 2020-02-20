#pragma once

#include <mirheo/core/domain.h>
#include <mirheo/core/datatypes.h>

#include <mirheo/core/utils/cpu_gpu_defines.h>
#include <mirheo/core/utils/helper_math.h>

namespace mirheo
{

class ParticleVector;

/// Represents a planar wall.
class StationaryWallPlane
{
public:
    /** \brief Construct a StationaryWallPlane
        \param [in] normal Normal of the wall, pointing inside the walls.
        \param [in] pointThrough One point inside the plane, in global coordinates.
     */
    StationaryWallPlane(real3 normal, real3 pointThrough) :
        normal_(normalize(normal)),
        pointThrough_(pointThrough)
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
    const StationaryWallPlane& handler() const
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
        const real dist = dot(normal_, gr - pointThrough_);
        return dist;
    }

private:
    real3 normal_;
    real3 pointThrough_;
    DomainInfo domain_;
};

} // namespace mirheo
