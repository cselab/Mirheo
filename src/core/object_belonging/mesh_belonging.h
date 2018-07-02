#pragma once

#include "interface.h"

class MeshBelongingChecker : public ObjectBelongingChecker
{
public:
	using ObjectBelongingChecker::ObjectBelongingChecker;

	void tagInner(ParticleVector* pv, CellList* cl, cudaStream_t stream) override;

	virtual ~MeshBelongingChecker() = default;
};
