#pragma once

#include <core/containers.h>

#include <string>

class ParticleVector;
class ObjectVector;
class CellList;

enum class BelongingTags
{
	Outside = 0, Inside, Boundary
};

class ObjectBelongingChecker
{
public:
	std::string name;

	ObjectBelongingChecker(std::string name) : name(name) { }

	/**
	 * Particle with tags == BelongingTags::Outside  will be copied to pvOut
	 *                    == BelongingTags::Inside   will be copied to pvIn
	 * Other particles are DROPPED (boundary particles)
	 */
	void splitByBelonging(ParticleVector* src, ParticleVector* pvIn, ParticleVector* pvOut, cudaStream_t stream);
	void checkInner(ParticleVector* pv, CellList* cl, cudaStream_t stream);

	virtual void setup(ObjectVector* ov) { this->ov = ov; }

	virtual ~ObjectBelongingChecker() = default;

protected:
	ObjectVector* ov;

	PinnedBuffer<BelongingTags> tags;
	PinnedBuffer<int> nInside{1}, nOutside{1};

	virtual void tagInner(ParticleVector* pv, CellList* cl, cudaStream_t stream) = 0;
};
