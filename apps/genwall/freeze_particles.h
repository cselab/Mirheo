#pragma once

class SDF_basedWall;
class ParticleVector;

void freezeParticlesInWall(SDF_basedWall* wall, ParticleVector* pv, ParticleVector* frozen, float minVal=0.0f, float maxVal=1.2f);

