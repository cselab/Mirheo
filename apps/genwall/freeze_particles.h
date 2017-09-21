#pragma once

class SDFWall;
class ParticleVector;

void freezeParticlesInWall(SDFWall* wall, ParticleVector* pv, ParticleVector* frozen, float minSdf=0.0f, float maxSdf=1.2f);
