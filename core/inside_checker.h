#pragma once

#include <core/containers.h>

class ParticleVector;
class ObjectVector;
class CellList;

/**
 * TAG >= 0 means that particle is definitely inside an object with id TAG
 * TAG == -1 means that particle is definitely outside
 * TAG == -2 means that particle is on the boundary
 */
class InsideChecker
{
private:
	ObjectVector* ov;

public:
	InsideChecker(ObjectVector* ov) : ov(ov) {};

	virtual void tagInner(ParticleVector* pv, CellList* cl, PinnedBuffer<int>& tags, int& nInside, int& nOutside, cudaStream_t stream) = 0;

	/**
	 * Particle with tags == -1 will be copied to pvOut
	 *                    >= 0  will be copied to pvIn
	 * Other particles are DROPPED
	 */
	static void splitByTags(ParticleVector* src, PinnedBuffer<int>& tags,
			int nInside, int nOutside, ParticleVector* pvIn, ParticleVector* pvOut, cudaStream_t stream);

	virtual ~InsideChecker() = default;
};

class EllipsoidInsideChecker : public InsideChecker
{
private:
	PinnedBuffer<int> nIn{1}, nOut{1};
	RigidEllipsoidObjectVector* rov;

public:
	void tagInner(ParticleVector* pv, CellList* cl, PinnedBuffer<int>& tags, int& nInside, int& nOutside, cudaStream_t stream) override;

	virtual ~EllipsoidInsideChecker() = default;
};
