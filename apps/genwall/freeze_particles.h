#pragma once

class Wall;
class ParticleVector;

void freezeParticlesInWall(Wall* wall, ParticleVector* pv, ParticleVector* frozen, float minSdf=0.0f, float maxSdf=1.2f);
