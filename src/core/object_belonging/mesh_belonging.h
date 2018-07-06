#pragma once

#include "object_belonging.h"

class MeshBelongingChecker : public ObjectBelongingChecker_Common
{
public:
	using ObjectBelongingChecker_Common::ObjectBelongingChecker_Common;

	void tagInner(ParticleVector* pv, CellList* cl, cudaStream_t stream) override;

	virtual ~MeshBelongingChecker() = default;
};
