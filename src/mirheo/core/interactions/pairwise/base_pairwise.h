// Copyright 2020 ETH Zurich. All Rights Reserved.
#pragma once

#include <mirheo/core/interactions/interface.h>
#include <mirheo/core/interactions/utils/parameters_wrap.h>

namespace mirheo
{

/** \brief Base class for short-range symmetric pairwise interactions
 */
class BasePairwiseInteraction : public Interaction
{
public:

    /** \brief Construct a base pairwise interaction from parameters.
        \param [in] state The global state of the system.
        \param [in] name The name of the interaction.
        \param [in] rc The cutoff radius of the interaction.
                       Must be positive and smaller than the sub-domain size.
    */
    BasePairwiseInteraction(const MirState *state, const std::string& name, real rc);

    ~BasePairwiseInteraction();

    /// \return the cut-off radius of the pairwise interaction.
    std::optional<real> getCutoffRadius() const override;

protected:
    real rc_; ///< cut-off radius of the interaction
};

} // namespace mirheo
