#pragma once

#include <core/containers.h>

#include <string>

class ParticleVector;
class ObjectVector;
class CellList;

/**
 * TAG >= 0 means that particle is definitely inside an object with id TAG
 * TAG == -1 means that particle is definitely outside
 * TAG == -2 means that particle is on the boundary
 */
class ObjectBelongingChecker
{
public:
	ObjectBelongingChecker(std::string name) : name(name) { }

	void checkInner(ParticleVector* pv, CellList* cl, cudaStream_t stream);

	/**
	 * Particle with tags == -1 will be copied to pvOut
	 *                    >= 0  will be copied to pvIn
	 * Other particles are DROPPED
	 */
	void splitByBelonging(ParticleVector* src, ParticleVector* pvIn, ParticleVector* pvOut, cudaStream_t stream);

	virtual ~ObjectBelongingChecker() = default;

protected:
	std::string name;

	ObjectVector* ov;

	PinnedBuffer<int> tags;
	PinnedBuffer<int> nInside{1}, nOutside{1};

	virtual void tagInner(ParticleVector* pv, CellList* cl, cudaStream_t stream) = 0;
};
