#pragma once

#include "base_pairwise.h"
#include "drivers.h"
#include "factory_helper.h"

#include <mirheo/core/celllist.h>
#include <mirheo/core/pvs/object_vector.h>
#include <mirheo/core/pvs/particle_vector.h>
#include <mirheo/core/pvs/views/pv.h>
#include <mirheo/core/snapshot.h>
#include <mirheo/core/utils/config.h>
#include <mirheo/core/utils/cuda_common.h>
#include <mirheo/core/utils/kernel_launch.h>

#include <fstream>
#include <map>

namespace mirheo
{

/** \brief Short-range symmetric pairwise interactions
    \tparam PairwiseKernel The functor that describes the interaction between two particles (interaction kernel).
    
    See the pairwise interaction entry of the developer documentation for the interface requirements of the kernel. 
 */
template <class PairwiseKernel>
class PairwiseInteraction : public BasePairwiseInteraction
{
public:
#ifndef DOXYGEN_SHOULD_SKIP_THIS // bug in breathe
    /// The parameters corresponding to the interaction kernel.
    using KernelParams = typename PairwiseKernel::ParamsType;
#endif // DOXYGEN_SHOULD_SKIP_THIS
    
    /** \brief Construct a PairwiseInteraction object
        \param [in] state The global state of the system
        \param [in] name The name of the interaction
        \param [in] rc The cut-off radius of the interaction
        \param [in] pairParams The parameters used to construct the interaction kernel
        \param [in] seed used to initialize random number generator (needed to construct some interaction kernels).
     */
    PairwiseInteraction(const MirState *state, const std::string& name, real rc,
                        KernelParams pairParams, long seed = 42424242) :
        BasePairwiseInteraction(state, name, rc),
        defaultPair_{rc, pairParams, state->dt, seed},
        _pairParams{pairParams}        
    {}
    
    /** \brief Constructs a PairwiseInteraction object from a snapshot.
        \param [in] state The global state of the system
        \param [in] loader The \c Loader object. Provides load context and unserialization functions.
        \param [in] config The parameters of the interaction.
     */
    PairwiseInteraction(const MirState *state, Loader& loader, const ConfigObject& config) :
        PairwiseInteraction(state, config["name"], config["rc"],
                            loader.load<KernelParams>(config["pairParams"]))
    {
        long seed = 42424242;
        warn("NOTE: Seed not serialized, resetting it!");
        const ConfigArray& mapArray = config["intMap"].getArray();
        if (mapArray.size() % 3 != 0) {
            die("The array's number of elements is supposed to be a "
                "multiple of 3, got %zu instead.", mapArray.size());
        }
        // intMap stored as an array [key1a, key1b, rawParams1, key2a, key2b, rawParams2, ...].
        for (size_t i = 0; i < mapArray.size(); i += 3) {
            KernelParams params = loader.load<KernelParams>(mapArray[i + 2]);
            intMap_.emplace(
                    std::make_pair(loader.load<std::string>(mapArray[i]),
                                   loader.load<std::string>(mapArray[i + 1])),
                    Kernel{PairwiseKernel{rc_, params, state->dt, seed}, params});
        }
    }
    
    ~PairwiseInteraction() = default;

    void setPrerequisites(ParticleVector *pv1, ParticleVector *pv2, CellList *cl1, CellList *cl2) override
    {
        if (outputsDensity <PairwiseKernel>::value ||
            requiresDensity<PairwiseKernel>::value   )
        {
            pv1->requireDataPerParticle<real>(channel_names::densities, DataManager::PersistenceMode::None);
            pv2->requireDataPerParticle<real>(channel_names::densities, DataManager::PersistenceMode::None);
            
            cl1->requireExtraDataPerParticle<real>(channel_names::densities);
            cl2->requireExtraDataPerParticle<real>(channel_names::densities);
        }
    }
    
    void local(ParticleVector *pv1, ParticleVector *pv2, CellList *cl1, CellList *cl2, cudaStream_t stream) override
    {
        // if (pv1->local()->size() < pv2->local()->size())
        _computeLocal(pv1, pv2, cl1, cl2, stream);
        // else
        //    computeLocal(pv2, pv1, cl2, cl1, state->currentTime, stream);
    }

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
            channels.push_back({channel_names::densities, Interaction::alwaysActive});

        return channels;
    }

    std::vector<InteractionChannel> getOutputChannels() const override
    {
        std::vector<InteractionChannel> channels;
        
        if (outputsDensity<PairwiseKernel>::value)
            channels.push_back({channel_names::densities, Interaction::alwaysActive});

        if (outputsForce<PairwiseKernel>::value)
            channels.push_back({channel_names::forces, Interaction::alwaysActive});

        return channels;
    }
    
    void setSpecificPair(const std::string& pv1name, const std::string& pv2name, const ParametersWrap::MapParams& mapParams) override
    {
        ParametersWrap desc(mapParams);
        auto params = _pairParams;
        factory_helper::readSpecificParams(params, desc);
        PairwiseKernel kernel {rc_, params, getState()->dt};
        _setSpecificPair(pv1name, pv2name, kernel, params);
    }

    void checkpoint(MPI_Comm comm, const std::string& path, int checkpointId) override
    {
        auto fname = createCheckpointNameWithId(path, "ParirwiseInt", "txt", checkpointId);
        {
            std::ofstream fout(fname);
            defaultPair_.writeState(fout);
            for (auto& entry : intMap_)
                entry.second.kernel.writeState(fout);
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
            check( entry.second.kernel.readState(fin) );
    }

    /// \return A string that describes the type of this object
    static std::string getTypeName()
    {
        return constructTypeName("PairwiseInteraction", 1, PairwiseKernel::getTypeName().c_str());
    }
    
    void saveSnapshotAndRegister(Saver& saver) override
    {
        saver.registerObject<PairwiseInteraction>(
                this, _saveSnapshot(saver, getTypeName()));
    }

protected:
    /** \brief Serialize raw parameters of all kernels.
        \param [in,out] saver The \c Saver object. Provides save context and serialization functions.
        \param [in] typeName The name of the type being saved.
    */
    ConfigObject _saveSnapshot(Saver& saver, const std::string& typeName)
    {
        ConfigObject config = BasePairwiseInteraction::_saveSnapshot(saver, typeName);
        config.emplace("pairParams", saver(_pairParams));

        // Key and value are stores as a list of three elements, two for the
        // key, one for the value (raw parameters).
        ConfigArray map;
        map.reserve(intMap_.size());
        for (const auto& pair : intMap_) {
            ConfigArray item;
            item.reserve(3);
            item.emplace_back(pair.first.first);   // Key.
            item.emplace_back(pair.first.second);  // Key.
            item.emplace_back(saver(pair.second.rawParams));  // Value (raw params).
            // TODO: Serialize RNG state.
            map.emplace_back(std::move(item));
        }
        config.emplace("intMap", std::move(map));
        return config;
    }

private:

    void _setSpecificPair(const std::string& pv1name, const std::string& pv2name, PairwiseKernel kernel, const KernelParams& rawParams)
    {
        intMap_.insert({{pv1name, pv2name}, {kernel, rawParams}});
        intMap_.insert({{pv2name, pv1name}, {kernel, rawParams}});
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
                CHOOSE_EXTERNAL(InteractionOutMode::NeedOutput, InteractionOutMode::NeedOutput, InteractionFetchMode::RowWise, pair.handler());
        }
    }

    /** \brief Compute halo forces
        
        Note: for ObjectVector objects, the forces will be computed even for halos.
        For pure ParticleVector objects, the halo forces are computed only locally (we rely on the pairwise force
        symetry for the neighbouring ranks). This avoids extra communications.
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
                CHOOSE_EXTERNAL(InteractionOutMode::NoOutput,   InteractionOutMode::NeedOutput, InteractionFetchMode::Dilute, pair.handler() );
            else
                CHOOSE_EXTERNAL(InteractionOutMode::NeedOutput, InteractionOutMode::NeedOutput, InteractionFetchMode::Dilute, pair.handler() );
    }

    PairwiseKernel& _getPairwiseKernel(const std::string& pv1name, const std::string& pv2name)
    {
        auto it = intMap_.find({pv1name, pv2name});
        if (it != intMap_.end())
        {
            debug("Using SPECIFIC parameters for PV pair '%s' -- '%s'", pv1name.c_str(), pv2name.c_str());
            return it->second.kernel;
        }
        else
        {
            debug("Using default parameters for PV pair '%s' -- '%s'", pv1name.c_str(), pv2name.c_str());
            return defaultPair_;
        }
    }

private:
    PairwiseKernel defaultPair_;
    KernelParams _pairParams;

    struct Kernel {
        PairwiseKernel kernel;
        KernelParams rawParams;
    };
    std::map< std::pair<std::string, std::string>, Kernel > intMap_;
};

} // namespace mirheo
