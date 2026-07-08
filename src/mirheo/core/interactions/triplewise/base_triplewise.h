// Copyright 2020 ETH Zurich. All Rights Reserved.
#pragma once

#include <mirheo/core/celllist.h>
#include <mirheo/core/interactions/interface.h>
#include <mirheo/core/interactions/utils/parameters_wrap.h>

#include <map>
#include <optional>

namespace mirheo
{

/** \brief Base class for short-range triplewise interactions

    Implementation detail:
        Triplewise interactions require halo particles of distance from
        boundary to up to 2*rc. To avoid extending Exchangers to support
        customized thickness, we instead override `getCutoffRadius` to return
        2*rc as an effective radius. This, however, degrades the performance of
        the kernel by a factor of (2x2x2)^2=64. To circumvent it, a copy of
        cell lists is stored, refined to the cell size of 1*rc.

        In principle, cell lists could be shared among all instances of
        BaseTriplewiseInteraction to enable reusing among multiple triplewise
        interactions. See InteractionManager and Simulation.
 */
class BaseTriplewiseInteraction : public Interaction
{
public:
    /** \brief Construct a base triplewise interaction from parameters.
        \param [in] state The global state of the system.
        \param [in] name The name of the interaction.
        \param [in] rc The cutoff radius of the interaction.
                       Must be positive and smaller than the sub-domain size.
    */
    BaseTriplewiseInteraction(const MirState *state, const std::string& name, real rc);
    ~BaseTriplewiseInteraction();

    /** Two-body variants of local() / halo(); must never be called on a
        triplewise interaction: they die. */
    void local(ParticleVector *pv1, ParticleVector *pv2,
               CellList *cl1, CellList *cl2, cudaStream_t stream) override;
    void halo(ParticleVector *pv1, ParticleVector *pv2, CellList *cl1,
              CellList *cl2, cudaStream_t stream) override;

    /// \return effective cutoff radius of 2*rc, to exchange 2*rc of ghost particles
    std::optional<real> getCutoffRadius() const override;

protected:
    /// Temporary cell lists for computing the interaction.
    struct CellListPair
    {
        CellListPair(ParticleVector *pv, real rc, const CellList *ref);

        CellList refinedLocal; ///< refined cells with size of 1*rc
        CellList halo;         ///< halo particles organized in cells with size of 1*rc
    };

    /** \brief Get or create the halo() and refined local() cell list.
        \param [in] pv \c ParticleVector to operate on
        \param [in] refCL Reference \c CellList with local particles
     */
    CellListPair* _getOrCreateCellLists(ParticleVector *pv, const CellList *refCL);

protected:
    /// cut-off radius of the interaction
    real rc_;

    /// cell lists for halo particles
    std::map<ParticleVector *, CellListPair> cellLists_;
};

} // namespace mirheo
