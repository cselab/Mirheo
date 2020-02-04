#include "simple_stationary_wall.h"

#pragma once

#include "interface.h"

#include <mirheo/core/containers.h>

namespace mirheo
{

class ParticleVector;
class CellList;


template<class InsideWallChecker, class VelocityField>
class WallWithVelocity : public SimpleStationaryWall<InsideWallChecker>
{
public:
    WallWithVelocity(const MirState *state, const std::string& name, InsideWallChecker&& insideWallChecker, VelocityField&& velField);

    void setup(MPI_Comm& comm) override;
    void attachFrozen(ParticleVector* pv) override;

    void bounce(cudaStream_t stream) override;

protected:
    VelocityField velField;
};

} // namespace mirheo
