// Copyright 2020 ETH Zurich. All Rights Reserved.
#pragma once

#include <mirheo/core/domain.h>
#include <mirheo/core/datatypes.h>

#include <mirheo/core/utils/cpu_gpu_defines.h>
#include <mirheo/core/utils/helper_math.h>
#include <mirheo/core/utils/reflection.h>

namespace mirheo
{

class ParticleVector;

/// Represents a sphere shape
class StationaryWallSphere
{
public:
    /** \brief Construct a StationaryWallSphere.
        \param [in] center Center of the sphere in global coordinates
        \param [in] radius Radius of the sphere
        \param [in] inside Domain is inside the box if set to \c true.
     */
    StationaryWallSphere(real3 center, real radius, bool inside) :
        center_(center),
        radius_(radius),
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
    const StationaryWallSphere& handler() const
    {
        return *this;
    }

    /** \brief Get the SDF of the current shape at a given position.
        \param [in] r position in local coordinates
        \return The SDF value
    */
    __D__ inline real operator()(real3 r) const
    {
        const real3 dr = domain_.local2global(r) - center_;
        const real dist = length(dr);

        return inside_ ? dist - radius_ : radius_ - dist;
    }

private:
    friend MemberVars<StationaryWallSphere>;

    real3 center_;
    real radius_;
    bool inside_;
    DomainInfo domain_;
};

MIRHEO_MEMBER_VARS(StationaryWallSphere, center_, radius_, inside_);

} // namespace mirheo
