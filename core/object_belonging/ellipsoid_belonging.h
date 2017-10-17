#pragma once

#include "interface.h"

class EllipsoidBelongingChecker : public ObjectBelongingChecker
{
public:
	using ObjectBelongingChecker::ObjectBelongingChecker;

	void tagInner(ParticleVector* pv, CellList* cl, cudaStream_t stream) override;

	virtual ~EllipsoidBelongingChecker() = default;
};
