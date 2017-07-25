#pragma once

class CellList;
class ParticleVector;
class RigidObjectVector;

void bounceFromRigidEllipsoid(ParticleVector* pv, CellList* cl, RigidObjectVector* rov, const float dt, bool local, cudaStream_t stream);
