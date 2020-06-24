// Copyright 2020 ETH Zurich. All Rights Reserved.
#pragma once

#include "simple_stationary_wall.h"

#include <mirheo/core/containers.h>

namespace mirheo
{

class ParticleVector;
class CellList;

/** \brief SDF-based wall with non zero velocity boundary conditions.
    \tparam InsideWallChecker Wall shape representation.
    \tparam VelocityField Wall velocity representation.
*/
template<class InsideWallChecker, class VelocityField>
class WallWithVelocity : public SimpleStationaryWall<InsideWallChecker>
{
public:
    /** \brief Construct a WallWithVelocity object.
        \param [in] state The simulation state.
        \param [in] name The wall name.
        \param [in] insideWallChecker A functor that represents the wall surface (see stationary_walls/).
        \param [in] velField A functor that represents the wall velocity (see velocity_field/).
     */
    WallWithVelocity(const MirState *state, const std::string& name, InsideWallChecker&& insideWallChecker, VelocityField&& velField);

    void setup(MPI_Comm& comm) override;
    void attachFrozen(ParticleVector* pv) override;

    void bounce(cudaStream_t stream) override;

private:
    VelocityField velField_; ///< The wall velocity field
};

} // namespace mirheo
