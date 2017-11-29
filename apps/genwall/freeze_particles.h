#pragma once

class Wall;
class ParticleVector;

void freezeParticlesWrapper(Wall* wall, ParticleVector* pv, ParticleVector* frozen, float minVal=0.0f, float maxVal=1.2f);

