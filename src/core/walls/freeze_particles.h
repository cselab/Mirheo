#pragma once

#include <vector>

class SDF_basedWall;
class ParticleVector;

void freezeParticlesInWall(SDF_basedWall *wall, ParticleVector *pv, float minVal=0.0f, float maxVal=1.2f);
void freezeParticlesInWalls(std::vector<SDF_basedWall*> walls, ParticleVector *pv, float minVal, float maxVal);
