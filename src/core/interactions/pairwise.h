#pragma once

#include "interface.h"
#include "pairwise/kernels/parameters.h"

class PairwiseInteraction : public Interaction
{
public:
    
    PairwiseInteraction(const MirState *state, const std::string& name, float rc,
                        VarPairwiseParams varParams, bool stress, float stressPeriod);
    ~PairwiseInteraction();

    void setPrerequisites(ParticleVector *pv1, ParticleVector *pv2, CellList *cl1, CellList *cl2) override;
    
    void local(ParticleVector *pv1, ParticleVector *pv2, CellList *cl1, CellList *cl2, cudaStream_t stream) override;
    void halo (ParticleVector *pv1, ParticleVector *pv2, CellList *cl1, CellList *cl2, cudaStream_t stream) override;

    Stage getStage() const override;

    std::vector<InteractionChannel> getInputChannels() const override;
    std::vector<InteractionChannel> getOutputChannels() const override;

    void checkpoint(MPI_Comm comm, const std::string& path, int checkpointId) override;
    void restart   (MPI_Comm comm, const std::string& path) override;

private:
    VarPairwiseParams varParams;
};
