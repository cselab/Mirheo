// Copyright 2020 ETH Zurich. All Rights Reserved.
#pragma once

#include "interface.h"

#include <mirheo/core/containers.h>

namespace mirheo
{
class LocalParticleVector;

/** Compute binding interaction used to attach two ParticleVector together.
    The interaction has the form of that of a linear spring with constant kBound.
 */
class ObjectBindingInteraction : public Interaction
{
public:

    /** Construct an ObjectBindingInteraction interaction.
        \param [in] state The global state of the system.
        \param [in] name The name of the interaction.
        \param [in] kBound The force coefficient (spring constant).
        \param [in] pairs The list of pairs of particles that will interact with each other.
                          A pair contains the global ids of the first ParticleVector (first entry)
                          and the second ParticleVector (second entry).
    */
    ObjectBindingInteraction(const MirState *state, std::string name,
                             real kBound, std::vector<int2> pairs);

    ~ObjectBindingInteraction();

    void local(ParticleVector *pv1, ParticleVector *pv2, CellList *cl1, CellList *cl2, cudaStream_t stream) override;
    void halo (ParticleVector *pv1, ParticleVector *pv2, CellList *cl1, CellList *cl2, cudaStream_t stream) override;

private:

    void _buildInteractionMap(ParticleVector *pv1, ParticleVector *pv2,
                              LocalParticleVector *lpv1, LocalParticleVector *lpv2,
                              cudaStream_t stream);

    void _computeForces(ParticleVector *pv1, ParticleVector *pv2,
                        LocalParticleVector *lpv1, LocalParticleVector *lpv2,
                        cudaStream_t stream) const;

private:

    real kBound_; ///< The spring constant
    DeviceBuffer<int2> pairs_; ///< Global ids of particles that interact.

    DeviceBuffer<int> gidToPv1Ids_; ///< helper map from global Id to local Ids of pv1.
    DeviceBuffer<int> gidToPv2Ids_; ///< helper map from global Id to local Ids of pv2.
    DeviceBuffer<int> partnersMaps_; ///< map from pv1 (local ids) to pv2 (local ids).
};

} // namespace mirheo
