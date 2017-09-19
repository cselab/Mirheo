#include "interface.h"

struct DummyIC : public InitialConditions
{
	void exec(const MPI_Comm& comm, ParticleVector* pv, float3 globalDomainStart, float3 localDomainSize, cudaStream_t stream) override
	{ }

	virtual ~DummyIC() = default;
};
