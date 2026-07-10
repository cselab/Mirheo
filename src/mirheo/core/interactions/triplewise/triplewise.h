// Copyright 2020 ETH Zurich. All Rights Reserved.
#pragma once

#include "base_triplewise.h"
#include "drivers.h"
#include "factory_helper.h"

#include <mirheo/core/celllist.h>
#include <mirheo/core/pvs/object_vector.h>
#include <mirheo/core/pvs/particle_vector.h>
#include <mirheo/core/pvs/views/pv.h>
#include <mirheo/core/utils/cuda_common.h>
#include <mirheo/core/utils/kernel_launch.h>

#include <fstream>
#include <map>

namespace mirheo
{

/** \brief Short-range symmetric triplewise interactions
    \tparam TriplewiseKernel The functor that describes the interaction between two particles (interaction kernel).

    See the triplewise interaction entry of the developer documentation for the interface requirements of the kernel.
 */
template <class TriplewiseKernel>
class TriplewiseInteraction : public BaseTriplewiseInteraction
{
public:
#ifndef DOXYGEN_SHOULD_SKIP_THIS // bug in breathe
    /// The parameters corresponding to the interaction kernel.
    using KernelParams = typename TriplewiseKernel::ParamsType;
#endif // DOXYGEN_SHOULD_SKIP_THIS

    /** \brief Construct a TriplewiseInteraction object
        \param [in] state The global state of the system
        \param [in] name The name of the interaction
        \param [in] rc The cut-off radius of the interaction
        \param [in] params The parameters used to construct the interaction kernel
        \param [in] seed used to initialize random number generator (needed to construct some interaction kernels).
     */
    TriplewiseInteraction(const MirState *state, const std::string& name, real rc,
                        KernelParams params, long seed = 42424242) :
        BaseTriplewiseInteraction(state, name, rc),
        kernel_{rc, params, seed}
    {}

    void setPrerequisites(ParticleVector *pv1, ParticleVector *pv2, ParticleVector *pv3,
                          CellList *cl1, CellList *cl2, CellList *cl3) override
    {
        if (outputsDensity <TriplewiseKernel>::value || requiresDensity<TriplewiseKernel>::value)
        {
            pv1->requireDataPerParticle<real>(channel_names::densities, DataManager::PersistenceMode::None);
            pv2->requireDataPerParticle<real>(channel_names::densities, DataManager::PersistenceMode::None);
            pv3->requireDataPerParticle<real>(channel_names::densities, DataManager::PersistenceMode::None);

            cl1->requireExtraDataPerParticle<real>(channel_names::densities);
            cl2->requireExtraDataPerParticle<real>(channel_names::densities);
            cl3->requireExtraDataPerParticle<real>(channel_names::densities);
        }
    }

    void local(ParticleVector *pv1, ParticleVector *pv2, ParticleVector *pv3,
               CellList *cl1, CellList *cl2, CellList *cl3, cudaStream_t stream) override
    {
        _computeLocal(pv1, pv2, pv3, cl1, cl2, cl3, stream);
    }

    void halo(ParticleVector *pv1, ParticleVector *pv2, ParticleVector *pv3,
              CellList *cl1, CellList *cl2, CellList *cl3, cudaStream_t stream) override
    {
        const bool isov1 = dynamic_cast<ObjectVector *>(pv1) != nullptr;
        const bool isov2 = dynamic_cast<ObjectVector *>(pv2) != nullptr;
        const bool isov3 = dynamic_cast<ObjectVector *>(pv3) != nullptr;

        if (isov1 || isov2 || isov3)
            die("3-body force with ObjectVectors not implemented.");
        if (pv1 != pv2 || pv2 != pv3 || pv3 != pv1)
            die("3-body forces with two or three different ParticleVectors not implemented.");

        CellList *clHalo = &_getOrCreateCellLists(pv1, cl1)->halo;
        clHalo->build(stream);

        (void)cl2;
        (void)cl3;
        // We use the original coarse cell lists here because
        //   a) `computeLocal` and `computeHalo` may run in parallel, we would
        //      need to rebuild `refinedLocal` cell list in the "Plugins:
        //      before forces" task or earlier.
        //   b) force accumulation from `refinedLocal` would have to be done
        //      after both `computeLocal` and `computeHalo` have completed,
        //      which again would require modifying the task schedule
        // Hence, we choose to use original `cl1`, at the cost of slowing down
        // the HLL step by 64x.
        _computeHalo111(pv1, cl1, clHalo, stream);
    }

    Stage getStage() const override
    {
        if (isFinal<TriplewiseKernel>::value)
            return Stage::Final;
        else
            return Stage::Intermediate;
    }

    std::vector<InteractionChannel> getInputChannels() const override
    {
        std::vector<InteractionChannel> channels;

        if (requiresDensity<TriplewiseKernel>::value)
            channels.push_back({channel_names::densities, Interaction::alwaysActive});

        return channels;
    }

    std::vector<InteractionChannel> getOutputChannels() const override
    {
        std::vector<InteractionChannel> channels;

        if (outputsDensity<TriplewiseKernel>::value)
            channels.push_back({channel_names::densities, Interaction::alwaysActive});

        if (outputsForce<TriplewiseKernel>::value)
            channels.push_back({channel_names::forces, Interaction::alwaysActive});

        return channels;
    }

    void checkpoint(MPI_Comm comm, const std::string& path, int checkpointId) override
    {
        auto fname = createCheckpointNameWithId(path, "TriplewiseInt", "txt", checkpointId);
        {
            std::ofstream fout(fname);
            kernel_.writeState(fout);
        }
        createCheckpointSymlink(comm, path, "TriplewiseInt", "txt", checkpointId);
    }

    void restart(__UNUSED MPI_Comm comm, const std::string& path) override
    {
        auto fname = createCheckpointName(path, "TriplewiseInt", "txt");
        std::ifstream fin(fname);

        auto check = [&](bool good) {
            if (!good) die("failed to read '%s'\n", fname.c_str());
        };

        check(fin.good());
        check( kernel_.readState(fin) );
    }

private:
    /** \brief Compute forces between all the triples of particles that are closer
        than rc to each other.
    */
    void _computeLocal(ParticleVector* pv1, ParticleVector* pv2, ParticleVector* pv3,
                       CellList* cl1, CellList* cl2, CellList* cl3, cudaStream_t stream)
    {
        using ViewType = typename TriplewiseKernel::ViewType;
        kernel_.setup(cl1, cl2, cl3, getState());

        /*  Self interaction */
        if (pv1 == pv2 && pv2 == pv3 && pv3 == pv1)
        {
            CellList *clRefined = &_getOrCreateCellLists(pv1, cl1)->refinedLocal;
            clRefined->build(stream);
            const auto view = clRefined->getView<ViewType>();
            if (view.size < 3)
                return;

            debug("Computing internal forces for %s (%d particles)", pv1->getCName(), view.size);

            // The refined cell list is not a part of InteractionManager, so we
            // manually clear and accumulate the forces.
            auto interactionChannels = getOutputChannels();
            std::vector<std::string> channelNames(interactionChannels.size());
            for (size_t i = 0; i < interactionChannels.size(); ++i)
                channelNames[i] = std::move(interactionChannels[i].name);
            clRefined->clearChannels(channelNames, stream);

            const int nth = 128;


            SAFE_KERNEL_LAUNCH(
                    LLL_,
                    getNblocks(view.size, nth), nth, 0, stream,
                    clRefined->cellInfo(), view, view, kernel_.handler());

            // Accumulate forces back to the particle vector.
            clRefined->accumulateChannelsAtomic(channelNames, stream);
        }
        else /*  External interaction */
        {
            die("3-body interactions with two or three different PVs not implemented.");
        }
    }

    /** \brief Compute halo forces for interactions within a single ParticleVector */
    void _computeHalo111(ParticleVector *pv, CellList *clLocal, CellList *clHalo, cudaStream_t stream)
    {
        using ViewType = typename TriplewiseKernel::ViewType;

        // Ask cell lists for the views, not the particle vector! Cell
        // lists store copies of the data, reordered according to cells.
        auto viewLocal = clLocal->getView<ViewType>();
        auto viewHalo = clHalo->getView<ViewType>();
        assert(viewLocal.size == pv->local()->size());
        assert(viewHalo.size == pv->halo()->size());
        debug("Computing internal forces for %s (%d local and %d halo particles)",
              pv->getCName(), viewLocal.size, viewHalo.size);

        if (viewLocal.size == 0 || viewHalo.size == 0 || viewLocal.size + viewHalo.size < 3)
            return;

        auto cinfoLocal = clLocal->cellInfo();
        auto cinfoHalo = clHalo->cellInfo();

        const int nth = 128;
        
        kernel_.setup(clHalo, clLocal, clLocal, getState());

        SAFE_KERNEL_LAUNCH(
                HLL_,
                getNblocks(viewHalo.size, nth), nth, 0, stream,
                cinfoLocal, viewHalo, viewLocal, kernel_.handler());
        
        kernel_.setup(clLocal, clHalo, clHalo, getState());

        SAFE_KERNEL_LAUNCH(
                LHH_,
                getNblocks(viewLocal.size, nth), nth, 0, stream,
                cinfoHalo, viewLocal, viewHalo, kernel_.handler());
    }

private:
    TriplewiseKernel kernel_;
};

} // namespace mirheo
