// Copyright 2020 ETH Zurich. All Rights Reserved.
#pragma once

#include <mirheo/core/interactions/interface.h>

namespace mirheo
{

/** \brief Base class to manage rod interactions

    Rod interactions must be used with a RodVector.
    They are internal forces, meaning that halo() does not compute anything.
 */
class BaseRodInteraction : public Interaction
{
public:
    /** \brief Construct a \c BaseRodInteraction
        \param [in] state The global state of the system
        \param [in] name Name of the interaction
    */
    BaseRodInteraction(const MirState *state, const std::string& name);
    ~BaseRodInteraction();

    void halo(ParticleVector *pv1, ParticleVector *pv2, CellList *cl1, CellList *cl2, cudaStream_t stream) final;

    bool isSelfObjectInteraction() const final;
};

} // namespace mirheo
