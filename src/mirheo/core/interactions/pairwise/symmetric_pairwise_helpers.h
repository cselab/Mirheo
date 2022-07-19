// Copyright 2020 ETH Zurich. All Rights Reserved.
#pragma once

#include "drivers.h"
#include "stress.h"

#include <mirheo/core/celllist.h>
#include <mirheo/core/pvs/object_vector.h>
#include <mirheo/core/pvs/particle_vector.h>
#include <mirheo/core/pvs/views/pv.h>
#include <mirheo/core/utils/cuda_common.h>
#include <mirheo/core/utils/kernel_launch.h>


namespace mirheo {
namespace symmetric_pairwise_helpers {

namespace details {

/** \brief  Convenience macro wrapper

    Select one of the available kernels for external interaction depending
    on the number of particles involved, report it and call
*/
#define DISPATCH_EXTERNAL(P1, P2, P3, TPP, INTERACTION_FUNCTION)        \
do{ debug2("Dispatched to "#TPP" thread(s) per particle variant");              \
    SAFE_KERNEL_LAUNCH(                                                         \
            computeExternalInteractions_##TPP##tpp<P1 COMMA P2 COMMA P3>,       \
            getNblocks(TPP*dstView.size, nth), nth, 0, stream,                  \
            dstView, cl2->cellInfo(), srcView, INTERACTION_FUNCTION); } while (0)

#define CHOOSE_EXTERNAL(P1, P2, P3, INTERACTION_FUNCTION)                                        \
    do{  if (dstView.size < 1000  ) { DISPATCH_EXTERNAL(P1, P2, P3, 27, INTERACTION_FUNCTION); } \
    else if (dstView.size < 10000 ) { DISPATCH_EXTERNAL(P1, P2, P3, 9,  INTERACTION_FUNCTION); } \
    else if (dstView.size < 400000) { DISPATCH_EXTERNAL(P1, P2, P3, 3,  INTERACTION_FUNCTION); } \
    else                            { DISPATCH_EXTERNAL(P1, P2, P3, 1,  INTERACTION_FUNCTION); } } while(0)


/** \brief Compute forces between all the pairs of particles that are closer
    than rc to each other.

    Depending on type and whether pv1 == pv2 call
    computeSelfInteractions() or computeExternalInteractions_1tpp()
    (or other variants of external interaction kernels).
*/
template<class PairwiseKernel>
void computeLocal(const MirState *state, PairwiseKernel& pair,
                  ParticleVector *pv1, ParticleVector *pv2,
                  CellList *cl1, CellList *cl2, cudaStream_t stream)
{
    using ViewType = typename PairwiseKernel::ViewType;

    pair.setup(pv1->local(), pv2->local(), cl1, cl2, state);

    /*  Self interaction */
    if (pv1 == pv2)
    {
        auto view = cl1->getView<ViewType>();
        const int np = view.size;
        debug("Computing internal forces for %s (%d particles)", pv1->getCName(), np);

        const int nth = 128;

        auto cinfo = cl1->cellInfo();
        SAFE_KERNEL_LAUNCH(
                           computeSelfInteractions,
                           getNblocks(np, nth), nth, 0, stream,
                           cinfo, view, pair.handler());
    }
    else /*  External interaction */
    {
        const int np1 = pv1->local()->size();
        const int np2 = pv2->local()->size();
        debug("Computing external forces for %s - %s (%d - %d particles)", pv1->getCName(), pv2->getCName(), np1, np2);

        auto dstView = cl1->getView<ViewType>();
        auto srcView = cl2->getView<ViewType>();

        const int nth = 128;
        if (np1 > 0 && np2 > 0)
        {
            CHOOSE_EXTERNAL(InteractionOutMode::NeedOutput,
                            InteractionOutMode::NeedOutput,
                            InteractionFetchMode::RowWise,
                            pair.handler());
        }
    }
}

/** \brief Compute halo forces

    Note: for ObjectVector objects, the forces will be computed even for halos, except when it's the same ObjectVector.
    For pure ParticleVector objects, the halo forces are computed only locally (we rely on the pairwise force
    symetry for the neighbouring ranks). This avoids extra communications.
*/
template<class PairwiseKernel>
void computeHalo(const MirState *state, PairwiseKernel& pair,
                 ParticleVector *pv1, ParticleVector *pv2,
                 CellList *cl1, CellList *cl2,
                 cudaStream_t stream)
{
    using ViewType = typename PairwiseKernel::ViewType;

    pair.setup(pv1->halo(), pv2->local(), cl1, cl2, state);

    const int np1 = pv1->halo()->size();  // note halo here
    const int np2 = pv2->local()->size();
    debug("Computing halo forces for %s(halo) - %s (%d - %d particles) with rc = %g",
          pv1->getCName(), pv2->getCName(), np1, np2, cl2->rc);

    ViewType dstView(pv1, pv1->halo());
    auto srcView = cl2->getView<ViewType>();

    const bool isOV1 = dynamic_cast<ObjectVector *>(pv1) != nullptr;

    const int nth = 128;

    if (np1 > 0 && np2 > 0)
    {
        if (!isOV1) // don't need forces for pure particle halo
        {
            CHOOSE_EXTERNAL(InteractionOutMode::NoOutput,
                            InteractionOutMode::NeedOutput,
                            InteractionFetchMode::Dilute,
                            pair.handler() );
        }
        else if (isOV1 && pv1 == pv2) // need to compute the forces only once when an object vector interacts with itself.
        {
            SAFE_KERNEL_LAUNCH(
                               computeExternalInteractionsSkipPairs_1tpp,
                               getNblocks(dstView.size, nth), nth, 0, stream,
                               dstView, cl2->cellInfo(), srcView, pair.handler());
        }
        else
        {
            CHOOSE_EXTERNAL(InteractionOutMode::NeedOutput,
                            InteractionOutMode::NeedOutput,
                            InteractionFetchMode::Dilute,
                            pair.handler() );
        }
    }
}

} // namespace details

template<class PairwiseKernel>
void computeLocalInteractions(const MirState *state, PairwiseKernel& pair,
                              ParticleVector *pv1, ParticleVector *pv2,
                              CellList *cl1, CellList *cl2,
                              cudaStream_t stream)
{
    details::computeLocal(state, pair, pv1, pv2, cl1, cl2, stream);
}

template<class PairwiseKernel>
void computeHaloInteractions(const MirState *state, PairwiseKernel& pair,
                             ParticleVector *pv1, ParticleVector *pv2,
                             CellList *cl1, CellList *cl2,
                             cudaStream_t stream)
{
    const bool isOV1 = dynamic_cast<ObjectVector *>(pv1) != nullptr;
    const bool isOV2 = dynamic_cast<ObjectVector *>(pv2) != nullptr;

    if (isOV1 && isOV2)
    {
        // Two object vectors. Compute just one interaction, doesn't matter which
        details::computeHalo(state, pair, pv1, pv2, cl1, cl2, stream);
    }
    else if (isOV1)
    {
        // One object vector. Compute just one interaction, with OV as the first argument
        details::computeHalo(state, pair, pv1, pv2, cl1, cl2, stream);
    }
    else if (isOV2)
    {
        // One object vector. Compute just one interaction, with OV as the first argument
        details::computeHalo(state, pair, pv2, pv1, cl2, cl1, stream);
    }
    else
    {
        // Both are particle vectors. Compute one interaction if pv1 == pv2 and two otherwise
        details::computeHalo(state, pair, pv1, pv2, cl1, cl2, stream);

        if (pv1 != pv2)
            details::computeHalo(state, pair, pv2, pv1, cl2, cl1, stream);
    }
}

} // namespace symmetric_pairwise_helpers



template<class PairwiseKernel>
void StressManager::computeLocalInteractions(const MirState *state,
                                             PairwiseKernel& pair,
                                             PairwiseStressWrapper<PairwiseKernel>& pairWithStress,
                                             ParticleVector *pv1, ParticleVector *pv2,
                                             CellList *cl1, CellList *cl2,
                                             cudaStream_t stream)
{
    const auto t = static_cast<real>(state->currentTime);

    if (lastStressTime_+stressPeriod_ <= t || lastStressTime_ == t)
    {
        symmetric_pairwise_helpers::computeLocalInteractions(state, pairWithStress, pv1, pv2, cl1, cl2, stream);
        lastStressTime_ = t;
    }
    else
    {
        symmetric_pairwise_helpers::computeLocalInteractions(state, pair, pv1, pv2, cl1, cl2, stream);
    }
}

template<class PairwiseKernel>
void StressManager::computeHaloInteractions(const MirState *state,
                                            PairwiseKernel& pair,
                                            PairwiseStressWrapper<PairwiseKernel>& pairWithStress,
                                            ParticleVector *pv1, ParticleVector *pv2,
                                            CellList *cl1, CellList *cl2,
                                            cudaStream_t stream)
{
    const auto t = static_cast<real>(state->currentTime);

    if (lastStressTime_+stressPeriod_ <= t || lastStressTime_ == t)
    {
        symmetric_pairwise_helpers::computeHaloInteractions(state, pairWithStress, pv1, pv2, cl1, cl2, stream);
        lastStressTime_ = t;
    }
    else
    {
        symmetric_pairwise_helpers::computeHaloInteractions(state, pair, pv1, pv2, cl1, cl2, stream);
    }
}


} // namespace mirheo
