#pragma once

#include <string>

class CellList;
class ParticleVector;
class ObjectVector;

class Bouncer
{
protected:
	virtual void exec(ObjectVector* ov, ParticleVector* pv, CellList* cl, float dt, cudaStream_t stream, bool local) = 0;

public:
	std::string name;

	Bouncer(std::string name) : name(name) {};

	void bounceLocal(ObjectVector* ov, ParticleVector* pv, CellList* cl, float dt, cudaStream_t stream) { exec(ov, pv, cl, dt, stream, true); }
	void bounceHalo (ObjectVector* ov, ParticleVector* pv, CellList* cl, float dt, cudaStream_t stream) { exec(ov, pv, cl, dt, stream, false); }

	virtual ~Bouncer() = default;
};
