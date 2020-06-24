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

    /** \brief Construct the interaction from a snapshot.
        \param [in] state The global state of the system.
        \param [in] loader The \c Loader object. Provides load context and unserialization functions.
        \param [in] config The parameters of the interaction.
     */
    BasePairwiseInteraction(const MirState *state, Loader& loader, const ConfigObject& config);
    ~BasePairwiseInteraction();

    /** \brief Convenience function that avoids having one BasePairwiseInteraction per pair of ParticleVector.
        \param [in] pv1name Name of one interacting ParticleVector
        \param [in] pv2name Name of the other interacting ParticleVector
        \param [in] mapParams Contains the parameters to modify.

        The order of pv1name and pv2name is not important.
        This method copies the parameters given in the constructor and of the object, modify only the values specified in mapParams,
        and store the new set of parameters for the pair of ParticleVector.
        These new parameters will be used when computing the forces for that pair of ParticleVector.
     */
    virtual void setSpecificPair(const std::string& pv1name, const std::string& pv2name, const ParametersWrap::MapParams& mapParams) = 0;

    /// \return the cut-off radius of the pairwise interaction.
    real getCutoffRadius() const override;

protected:
    /** \brief Snapshot saving for base pairwise interactions. Stores the cutoff value.
        \param [in,out] saver The \c Saver object. Provides save context and serialization functions.
        \param [in] typeName The name of the type being saved.
    */
    ConfigObject _saveSnapshot(Saver& saver, const std::string& typeName);

protected:
    real rc_; ///< cut-off radius of the interaction
};

} // namespace mirheo
