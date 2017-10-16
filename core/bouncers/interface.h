#pragma once

#include <string>

class CellList;
class ParticleVector;
class ObjectVector;

class Bouncer
{
protected:
	ObjectVector* ov;

	virtual void exec (ParticleVector* pv, CellList* cl, float dt, cudaStream_t stream, bool local) = 0;

public:
	std::string name;

	Bouncer(std::string name) : name(name) {};

	virtual void setup(ObjectVector* ov) = 0;

	void bounceLocal(ParticleVector* pv, CellList* cl, float dt, cudaStream_t stream) { exec (pv, cl, dt, stream, true);  }
	void bounceHalo (ParticleVector* pv, CellList* cl, float dt, cudaStream_t stream) { exec (pv, cl, dt, stream, false); }

	virtual ~Bouncer() = default;
};
