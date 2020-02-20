#pragma once

#include <mirheo/core/interactions/interface.h>

namespace mirheo
{

class MembraneVector;

/** \brief Base class that represents membrane interactions.

    This kind of interactions does not require any cell-lists and is always a "self-interaction", 
    hence the halo interaction does not do anything.
    This must be used only with MembraneVector objects.
 */
class BaseMembraneInteraction : public Interaction
{
public:
    /** \brief Construct a BaseMembraneInteraction object
        \param [in] state The global state of the system
        \param [in] name The name of the interaction
    */
    BaseMembraneInteraction(const MirState *state, const std::string& name);

    /** \brief Construct the object from a snapshot
        \param [in] state The global state of the system
        \param [in] loader The \c Loader object. Provides load context and unserialization functions.
        \param [in] config The parameters of the interaction.
    */
    BaseMembraneInteraction(const MirState *state, Loader& loader, const ConfigObject& config);
    ~BaseMembraneInteraction();

    /** \brief Set the required channels to the concerned ParticleVector that will participate in the interactions.
        \param [in] pv1 The conserned data that will participate in the interactions.
        \param [in] pv2 The conserned data that will participate in the interactions.
        \param cl1 Unused
        \param cl2 Unused

        This method will fail if pv1 is not a MembraneVector or if pv1 is not the same as pv2. 
    */
    void setPrerequisites(ParticleVector *pv1, ParticleVector *pv2, CellList *cl1, CellList *cl2) override;

    void halo(ParticleVector *pv1, ParticleVector *pv2, CellList *cl1, CellList *cl2, cudaStream_t stream) final;

    bool isSelfObjectInteraction() const override;
    
protected:

    /** \brief Compute quantities used inside the force kernels.
        \param [in,out] mv the MembraneVector that will store the precomputed quantities
        \param [in] stream Stream used for the kernel executions.

        Must be called before every force kernel.
        default: compute area and volume of each cell
     */
    virtual void _precomputeQuantities(MembraneVector *mv, cudaStream_t stream);
};

} // namespace mirheo
