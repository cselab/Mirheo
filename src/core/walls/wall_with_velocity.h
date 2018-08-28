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
    WallWithVelocity(std::string name, InsideWallChecker&& insideWallChecker, VelocityField&& velField) :
        SimpleStationaryWall<InsideWallChecker>(name, std::move(insideWallChecker)),
        velField(std::move(velField))
    {    }

    void setup(MPI_Comm& comm, DomainInfo domain) override;
    void attachFrozen(ParticleVector* pv) override;

    void bounce(float dt, cudaStream_t stream) override;

protected:
    VelocityField velField;
    DomainInfo domain;
};
