#include "simple_stationary_wall.h"

#pragma once

#include "interface.h"

#include <core/containers.h>

class ParticleVector;
class CellList;


template<class InsideWallChecker, class VelocityField>
class WallWithVelocity : public SimpleStationaryWall<InsideWallChecker>
{
public:
    WallWithVelocity(std::string name, const MirState *state, InsideWallChecker&& insideWallChecker, VelocityField&& velField);

    void setup(MPI_Comm& comm) override;
    void attachFrozen(ParticleVector* pv) override;

    void bounce(cudaStream_t stream) override;

protected:
    VelocityField velField;
};
