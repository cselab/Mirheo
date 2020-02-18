#pragma once

#include "interface.h"
#include "membrane/filters/api.h"
#include "membrane/kernels/parameters.h"

#include <extern/variant/include/mpark/variant.hpp>

#include <memory>

namespace mirheo
{

using VarBendingParams = mpark::variant<KantorBendingParameters, JuelicherBendingParameters>;
using VarShearParams   = mpark::variant<WLCParameters, LimParameters>;

/**
 * membrane interactions
 * forces depend on the passed variant parameters
 */
class MembraneInteraction : public Interaction
{
public:

    MembraneInteraction(const MirState *state, std::string name, CommonMembraneParameters commonParams,
                        VarBendingParams varBendingParams, VarShearParams varShearParams,
                        bool stressFree, real growUntil, VarMembraneFilter varFilter);

    /** \brief Construct the interaction from a snapshot.
        \param [in] state The global state of the system.
        \param [in] loader The \c Loader object. Provides load context and unserialization functions.
        \param [in] config The parameters of the interaction.
     */
    MembraneInteraction(const MirState *state, Loader& loader, const ConfigObject& config);

    ~MembraneInteraction();

    /** \brief Create a \c ConfigObject describing the interaction and register it in the saver.
        \param [in,out] saver The \c Saver object. Provides save context and serialization functions.

        Checks that the object type is exactly \c MembraneInteraction.
      */
    void saveSnapshotAndRegister(Saver& saver) override;
    
    void setPrerequisites(ParticleVector *pv1, ParticleVector *pv2, CellList *cl1, CellList *cl2) override;

    void local (ParticleVector *pv1, ParticleVector *pv2, CellList *cl1, CellList *cl2, cudaStream_t stream) final;
    void halo  (ParticleVector *pv1, ParticleVector *pv2, CellList *cl1, CellList *cl2, cudaStream_t stream) final;

    bool isSelfObjectInteraction() const override;
    
protected:

    /**
     * compute quantities used by the force kernels.
     * this is called before every force kernel (see implementation of @ref local)
     * default: compute area and volume of each cell
     */
    virtual void precomputeQuantities(ParticleVector *pv1, cudaStream_t stream);
};

} // namespace mirheo
