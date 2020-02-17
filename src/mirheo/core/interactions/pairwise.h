#pragma once

#include "interface.h"
#include "pairwise/kernels/parameters.h"
#include "parameters_wrap.h"

namespace mirheo
{

class PairwiseInteraction : public Interaction
{
public:
    
    PairwiseInteraction(const MirState *state, const std::string& name, real rc,
                        const VarPairwiseParams& varParams, const VarStressParams& varStressParams);

    /** \brief Construct the interaction from a snapshot.
        \param [in] state The global state of the system.
        \param [in] loader The \c Loader object. Provides load context and unserialization functions.
        \param [in] config The parameters of the interaction.
     */
    PairwiseInteraction(const MirState *state, Loader& loader, const ConfigObject& config);

    ~PairwiseInteraction();

    void setPrerequisites(ParticleVector *pv1, ParticleVector *pv2, CellList *cl1, CellList *cl2) override;
    
    void local(ParticleVector *pv1, ParticleVector *pv2, CellList *cl1, CellList *cl2, cudaStream_t stream) override;
    void halo (ParticleVector *pv1, ParticleVector *pv2, CellList *cl1, CellList *cl2, cudaStream_t stream) override;

    Stage getStage() const override;

    std::vector<InteractionChannel> getInputChannels() const override;
    std::vector<InteractionChannel> getOutputChannels() const override;

    void checkpoint(MPI_Comm comm, const std::string& path, int checkpointId) override;
    void restart   (MPI_Comm comm, const std::string& path) override;

    /** \brief Create a \c ConfigObject describing the interaction and register it in the saver.
        \param [in,out] saver The \c Saver object. Provides save context and serialization functions.

        Checks that the object type is exactly \c PairwiseInteraction.
      */
    void saveSnapshotAndRegister(Saver& saver) override;

    void setSpecificPair(ParticleVector *pv1, ParticleVector *pv2, const ParametersWrap::MapParams& desc);

    real getCutoffRadius() const override;

protected:
    /** \brief Implementation of snapshot saving. Reusable by potential derived classes.
        \param [in,out] saver The \c Saver object. Provides save context and serialization functions.
        \param [in] typeName The name of the type being saved.
      */
    ConfigObject _saveSnapshot(Saver& saver, const std::string& typeName);

private:
    VarPairwiseParams varParams_;
    VarStressParams varStressParams_;
};

} // namespace mirheo
