#pragma once

#include "base_pairwise.h"
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

/** \brief Short-range symmetric pairwise interactions
 */
template <class PairwiseKernel>
class PairwiseInteraction : public BasePairwiseInteraction
{
public:
    
    PairwiseInteraction(const MirState *state, const std::string& name, real rc,
                        typename PairwiseKernel::ParamsType pairParams, long seed = 42424242) :
        BasePairwiseInteraction(state, name, rc),
        defaultPair_{rc, pairParams, state->dt, seed},
        _pairParams{pairParams}        
    {}
    
    ~PairwiseInteraction() = default;

    void setPrerequisites(ParticleVector *pv1, ParticleVector *pv2, CellList *cl1, CellList *cl2) override
    {
        if (outputsDensity <PairwiseKernel>::value ||
            requiresDensity<PairwiseKernel>::value   )
        {
            pv1->requireDataPerParticle<real>(ChannelNames::densities, DataManager::PersistenceMode::None);
            pv2->requireDataPerParticle<real>(ChannelNames::densities, DataManager::PersistenceMode::None);
            
            cl1->requireExtraDataPerParticle<real>(ChannelNames::densities);
            cl2->requireExtraDataPerParticle<real>(ChannelNames::densities);
        }
    }
    
    /// Interface to computeLocal().
    void local(ParticleVector *pv1, ParticleVector *pv2, CellList *cl1, CellList *cl2, cudaStream_t stream) override
    {
        // if (pv1->local()->size() < pv2->local()->size())
        _computeLocal(pv1, pv2, cl1, cl2, stream);
        // else
        //    computeLocal(pv2, pv1, cl2, cl1, state->currentTime, stream);
    }

    /** \brief Interface to computeHalo().
     */
    void halo(ParticleVector *pv1, ParticleVector *pv2, CellList *cl1, CellList *cl2, cudaStream_t stream) override
    {
        const bool isov1 = dynamic_cast<ObjectVector *>(pv1) != nullptr;
        const bool isov2 = dynamic_cast<ObjectVector *>(pv2) != nullptr;

        if (isov1 && isov2)
        {
             // Two object vectors. Compute just one interaction, doesn't matter which
            _computeHalo(pv1, pv2, cl1, cl2, stream);
        }
        else if (isov1)
        {
            // One object vector. Compute just one interaction, with OV as the first argument
            _computeHalo(pv1, pv2, cl1, cl2, stream);
        }
        else if (isov2)
        {
            // One object vector. Compute just one interaction, with OV as the first argument
            _computeHalo(pv2, pv1, cl2, cl1, stream);
        }
        else
        {
            // Both are particle vectors. Compute one interaction if pv1 == pv2 and two otherwise
            _computeHalo(pv1, pv2, cl1, cl2, stream);
            if (pv1 != pv2)
                _computeHalo(pv2, pv1, cl2, cl1, stream);
        }
    }

    Stage getStage() const override
    {
        if (isFinal<PairwiseKernel>::value)
            return Stage::Final;
        else
            return Stage::Intermediate;
    }

    std::vector<InteractionChannel> getInputChannels() const override
    {
        std::vector<InteractionChannel> channels;
        
        if (requiresDensity<PairwiseKernel>::value)
            channels.push_back({ChannelNames::densities, Interaction::alwaysActive});

        return channels;
    }

    std::vector<InteractionChannel> getOutputChannels() const override
    {
        std::vector<InteractionChannel> channels;
        
        if (outputsDensity<PairwiseKernel>::value)
            channels.push_back({ChannelNames::densities, Interaction::alwaysActive});

        if (outputsForce<PairwiseKernel>::value)
            channels.push_back({ChannelNames::forces, Interaction::alwaysActive});

        return channels;
    }
    
    void setSpecificPair(const std::string& pv1name, const std::string& pv2name, const ParametersWrap::MapParams& mapParams) override
    {
        ParametersWrap desc(mapParams);
        auto params = _pairParams;
        FactoryHelper::readSpecificParams(params, desc);
        PairwiseKernel kernel {rc_, params, getState()->dt};
        _setSpecificPair(pv1name, pv2name, kernel);
    }

    void checkpoint(MPI_Comm comm, const std::string& path, int checkpointId) override
    {
        auto fname = createCheckpointNameWithId(path, "ParirwiseInt", "txt", checkpointId);
        {
            std::ofstream fout(fname);
            defaultPair_.writeState(fout);
            for (auto& entry : intMap_)
                entry.second.writeState(fout);
        }
        createCheckpointSymlink(comm, path, "ParirwiseInt", "txt", checkpointId);
    }
    
    void restart(__UNUSED MPI_Comm comm, const std::string& path) override
    {
        auto fname = createCheckpointName(path, "ParirwiseInt", "txt");
        std::ifstream fin(fname);

        auto check = [&](bool good) {
            if (!good) die("failed to read '%s'\n", fname.c_str());
        };

        check(fin.good());
        
        check( defaultPair_.readState(fin) );
        for (auto& entry : intMap_)
            check( entry.second.readState(fin) );
    }

private:

    void _setSpecificPair(const std::string& pv1name, const std::string& pv2name, PairwiseKernel pair)
    {
        intMap_.insert({{pv1name, pv2name}, pair});
        intMap_.insert({{pv2name, pv1name}, pair});
    }

    /** \brief  Convenience macro wrapper
     
        Select one of the available kernels for external interaction depending
        on the number of particles involved, report it and call
     */
    #define DISPATCH_EXTERNAL(P1, P2, P3, TPP, INTERACTION_FUNCTION)                \
    do{ debug2("Dispatched to "#TPP" thread(s) per particle variant");              \
        SAFE_KERNEL_LAUNCH(                                                         \
                computeExternalInteractions_##TPP##tpp<P1 COMMA P2 COMMA P3>,       \
                getNblocks(TPP*dstView.size, nth), nth, 0, stream,                  \
                dstView, cl2->cellInfo(), srcView, rc_*rc_, INTERACTION_FUNCTION); } while (0)

    #define CHOOSE_EXTERNAL(P1, P2, P3, INTERACTION_FUNCTION)                                        \
        do{  if (dstView.size < 1000  ) { DISPATCH_EXTERNAL(P1, P2, P3, 27, INTERACTION_FUNCTION); } \
        else if (dstView.size < 10000 ) { DISPATCH_EXTERNAL(P1, P2, P3, 9,  INTERACTION_FUNCTION); } \
        else if (dstView.size < 400000) { DISPATCH_EXTERNAL(P1, P2, P3, 3,  INTERACTION_FUNCTION); } \
        else                            { DISPATCH_EXTERNAL(P1, P2, P3, 1,  INTERACTION_FUNCTION); } } while(0)


    /** \brief Compute forces between all the pairs of particles that are closer
        than #rc to each other.
     
        Depending on type and whether pv1 == pv2 call
        computeSelfInteractions() or computeExternalInteractions_1tpp()
        (or other variants of external interaction kernels).
    */
    void _computeLocal(ParticleVector* pv1, ParticleVector* pv2, CellList* cl1, CellList* cl2, cudaStream_t stream)
    {
        auto& pair = _getPairwiseKernel(pv1->getName(), pv2->getName());
        using ViewType = typename PairwiseKernel::ViewType;

        pair.setup(pv1->local(), pv2->local(), cl1, cl2, getState());

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
                 cinfo, view, rc_*rc_, pair.handler());
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
                CHOOSE_EXTERNAL(InteractionOut::NeedAcc, InteractionOut::NeedAcc, InteractionMode::RowWise, pair.handler());
        }
    }

    /** \brief Compute halo forces
     */
    void _computeHalo(ParticleVector *pv1, ParticleVector *pv2, CellList *cl1, CellList *cl2, cudaStream_t stream)
    {
        auto& pair = _getPairwiseKernel(pv1->getName(), pv2->getName());
        using ViewType = typename PairwiseKernel::ViewType;

        pair.setup(pv1->halo(), pv2->local(), cl1, cl2, getState());

        const int np1 = pv1->halo()->size();  // note halo here
        const int np2 = pv2->local()->size();
        debug("Computing halo forces for %s(halo) - %s (%d - %d particles) with rc = %g", pv1->getCName(), pv2->getCName(), np1, np2, cl2->rc);

        ViewType dstView(pv1, pv1->halo());
        auto srcView = cl2->getView<ViewType>();
        
        const int nth = 128;
        if (np1 > 0 && np2 > 0)
            if (dynamic_cast<ObjectVector*>(pv1) == nullptr) // don't need forces for pure particle halo
                CHOOSE_EXTERNAL(InteractionOut::NoAcc,   InteractionOut::NeedAcc, InteractionMode::Dilute, pair.handler() );
            else
                CHOOSE_EXTERNAL(InteractionOut::NeedAcc, InteractionOut::NeedAcc, InteractionMode::Dilute, pair.handler() );
    }

    PairwiseKernel& _getPairwiseKernel(const std::string& pv1name, const std::string& pv2name)
    {
        auto it = intMap_.find({pv1name, pv2name});
        if (it != intMap_.end())
        {
            debug("Using SPECIFIC parameters for PV pair '%s' -- '%s'", pv1name.c_str(), pv2name.c_str());
            return it->second;
        }
        else
        {
            debug("Using default parameters for PV pair '%s' -- '%s'", pv1name.c_str(), pv2name.c_str());
            return defaultPair_;
        }
    }

private:
    PairwiseKernel defaultPair_;
    typename PairwiseKernel::ParamsType _pairParams;
    std::map< std::pair<std::string, std::string>, PairwiseKernel > intMap_;
};

} // namespace mirheo
