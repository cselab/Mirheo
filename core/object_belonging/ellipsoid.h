#pragma once

#include "interface.h"

class EllipsoidBelongingChecker : public ObjectBelongingChecker
{
public:
	void tagInner(ParticleVector* pv, CellList* cl, PinnedBuffer<int>& tags, int& nInside, int& nOutside, cudaStream_t stream) override;

	virtual ~EllipsoidBelongingChecker() = default;
};
