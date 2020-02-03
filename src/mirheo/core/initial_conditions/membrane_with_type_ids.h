#pragma once

#include "membrane.h"

namespace mirheo
{

class MembraneWithTypeIdsIC : public MembraneIC
{
public:
    MembraneWithTypeIdsIC(const std::vector<ComQ>& comQ, const std::vector<int>& typeIds, real globalScale = 1.0_r);
    ~MembraneWithTypeIdsIC();

    void exec(const MPI_Comm& comm, ParticleVector *pv, cudaStream_t stream) override;

private:
    std::vector<int> typeIds_;
};

} // namespace mirheo
