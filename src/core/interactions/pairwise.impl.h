#pragma once

#include "interface.h"

#include "pairwise/kernels.h"

#include <core/celllist.h>
#include <core/pvs/object_vector.h>
#include <core/pvs/particle_vector.h>
#include <core/pvs/views/pv.h>
#include <core/utils/cuda_common.h>
#include <core/utils/kernel_launch.h>

#include <fstream>
#include <map>

/**
 * Implementation of short-range symmetric pairwise interactions
 */
template<class PairwiseInteraction>
class InteractionPair : public Interaction
{
public:
    
    InteractionPair(const MirState *state, std::string name, float rc, PairwiseInteraction pair) :
        Interaction(state, name, rc),
        defaultPair(pair)
    {}
    
    ~InteractionPair() = default;

    /**
     * Interface to computeLocal().
     */
    void local(ParticleVector *pv1, ParticleVector *pv2, CellList *cl1, CellList *cl2, cudaStream_t stream) override
    {
        // if (pv1->local()->size() < pv2->local()->size())
        computeLocal(pv1, pv2, cl1, cl2, stream);
        // else
        //    computeLocal(pv2, pv1, cl2, cl1, state->currentTime, stream);
    }

    /**
     * Interface to computeHalo().
     *
     * The following cases exist:
     * - If one of \p pv1 or \p pv2 is ObjectVector, then only call to the _compute()
     *   needed: for halo ObjectVector another ParticleVector (or ObjectVector).
     *   This is because ObjectVector will collect the forces from remote processors,
     *   so we don't need to compute them twice.
     *
     * - Both are ParticleVector. Then if they are different, two _compute() calls
     *   are made such that halo1 \<-\> local2 and halo2 \<-\> local1. If \p pv1 and
     *   \p pv2 are the same, only one call is needed
     */
    void halo(ParticleVector *pv1, ParticleVector *pv2, CellList *cl1, CellList *cl2, cudaStream_t stream) override
    {
        auto isov1 = dynamic_cast<ObjectVector *>(pv1) != nullptr;
        auto isov2 = dynamic_cast<ObjectVector *>(pv2) != nullptr;

        // Two object vectors. Compute just one interaction, doesn't matter which
        if (isov1 && isov2) {
            computeHalo(pv1, pv2, cl1, cl2, stream);
            return;
        }

        // One object vector. Compute just one interaction, with OV as the first
        // argument
        if (isov1) {
            computeHalo(pv1, pv2, cl1, cl2, stream);
            return;
        }

        if (isov2) {
            computeHalo(pv2, pv1, cl2, cl1, stream);
            return;
        }

        // Both are particle vectors. Compute one interaction if pv1 == pv2 and two
        // otherwise
        computeHalo(pv1, pv2, cl1, cl2, stream);
        if (pv1 != pv2)
            computeHalo(pv2, pv1, cl2, cl1, stream);
    }
    
    void setSpecificPair(std::string pv1name, std::string pv2name, PairwiseInteraction pair)
    {
        intMap.insert({{pv1name, pv2name}, pair});
        intMap.insert({{pv2name, pv1name}, pair});
    }

    void checkpoint(MPI_Comm comm, std::string path, int checkpointId) override
    {
        auto fname = createCheckpointNameWithId(path, "ParirwiseInt", "txt", checkpointId);
        {
            std::ofstream fout(fname);
            defaultPair.writeState(fout);
            for (auto& entry : intMap)
                entry.second.writeState(fout);
        }
        createCheckpointSymlink(comm, path, "ParirwiseInt", "txt", checkpointId);
    }
    
    void restart(__UNUSED MPI_Comm comm, std::string path) override
    {
        auto fname = createCheckpointName(path, "ParirwiseInt", "txt");
        std::ifstream fin(fname);

        auto check = [&](bool good) {
            if (!good) die("failed to read '%s'\n", fname.c_str());
        };

        check(fin.good());
        
        check( defaultPair.readState(fin) );
        for (auto& entry : intMap)
            check( entry.second.readState(fin) );
    }

private:

    PairwiseInteraction defaultPair;
    std::map< std::pair<std::string, std::string>, PairwiseInteraction > intMap;

private:

    /**
     * Convenience macro wrapper
     *
     * Select one of the available kernels for external interaction depending
     * on the number of particles involved, report it and call
     */
    #define DISPATCH_EXTERNAL(P1, P2, P3, TPP, INTERACTION_FUNCTION)                \
    do{ debug2("Dispatched to "#TPP" thread(s) per particle variant");              \
        SAFE_KERNEL_LAUNCH(                                                         \
                computeExternalInteractions_##TPP##tpp<P1 COMMA P2 COMMA P3>,       \
                getNblocks(TPP*dstView.size, nth), nth, 0, stream,                  \
                dstView, cl2->cellInfo(), srcView, rc*rc, INTERACTION_FUNCTION); } while (0)

    #define CHOOSE_EXTERNAL(P1, P2, P3, INTERACTION_FUNCTION)                                        \
        do{  if (dstView.size < 1000  ) { DISPATCH_EXTERNAL(P1, P2, P3, 27, INTERACTION_FUNCTION); } \
        else if (dstView.size < 10000 ) { DISPATCH_EXTERNAL(P1, P2, P3, 9,  INTERACTION_FUNCTION); } \
        else if (dstView.size < 400000) { DISPATCH_EXTERNAL(P1, P2, P3, 3,  INTERACTION_FUNCTION); } \
        else                            { DISPATCH_EXTERNAL(P1, P2, P3, 1,  INTERACTION_FUNCTION); } } while(0)


    /**
     * Compute forces between all the pairs of particles that are closer
     * than #rc to each other.
     *
     * Depending on \p type and whether \p pv1 == \p pv2 call
     * computeSelfInteractions() or computeExternalInteractions_1tpp()
     * (or other variants of external interaction kernels).
     *
     * @tparam PariwiseInteraction is a functor that computes the force
     * given a pair of particles. It has to
     * provide two functions:
     * - This function will be called once before interactions computation
     *   and allows the functor to obtain required variables or data
     *   channels from the two ParticleVector and CellList:
     *   \code setup(ParticleVector* pv1, ParticleVector* pv2, CellList* cl1, CellList* cl2, float t) \endcode
     *
     * - This should be a \c \_\_device\_\_ operator that computes
     *   the force. It will be called for each close enough particle pair:
     *   \code float3 operator()(const Particle dst, int dstId, const Particle src, int srcId) const \endcode
     *   Return value of that call is force acting on the first particle,
     *   force acting on the second one is just opposite.
     */
    void computeLocal(ParticleVector* pv1, ParticleVector* pv2, CellList* cl1, CellList* cl2, cudaStream_t stream)
    {
        auto& pair = getPairwiseInteraction(pv1->name, pv2->name);
        using ViewType = typename PairwiseInteraction::ViewType;

        pair.setup(pv1->local(), pv2->local(), cl1, cl2, state);

        /*  Self interaction */
        if (pv1 == pv2)
        {
            auto view = cl1->getView<ViewType>();
            const int np = view.size;
            debug("Computing internal forces for %s (%d particles)", pv1->name.c_str(), np);

            const int nth = 128;

            auto cinfo = cl1->cellInfo();
            SAFE_KERNEL_LAUNCH(
                               computeSelfInteractions,
                               getNblocks(np, nth), nth, 0, stream,
                               cinfo, view, rc*rc, pair.handler());
        }
        else /*  External interaction */
        {
            const int np1 = pv1->local()->size();
            const int np2 = pv2->local()->size();
            debug("Computing external forces for %s - %s (%d - %d particles)", pv1->name.c_str(), pv2->name.c_str(), np1, np2);

            auto dstView = cl1->getView<ViewType>();
            auto srcView = cl2->getView<ViewType>();

            const int nth = 128;
            if (np1 > 0 && np2 > 0)
                CHOOSE_EXTERNAL(InteractionOut::NeedAcc, InteractionOut::NeedAcc, InteractionMode::RowWise, pair.handler());
        }
    }

    /**
     * Compute halo forces
     */
    void computeHalo(ParticleVector *pv1, ParticleVector *pv2, CellList *cl1, CellList *cl2, cudaStream_t stream)
    {
        auto& pair = getPairwiseInteraction(pv1->name, pv2->name);
        using ViewType = typename PairwiseInteraction::ViewType;

        pair.setup(pv1->halo(), pv2->local(), cl1, cl2, state);

        const int np1 = pv1->halo()->size();  // note halo here
        const int np2 = pv2->local()->size();
        debug("Computing halo forces for %s(halo) - %s (%d - %d particles) with rc = %g", pv1->name.c_str(), pv2->name.c_str(), np1, np2, cl2->rc);

        ViewType dstView(pv1, pv1->halo());
        auto srcView = cl2->getView<ViewType>();
        
        const int nth = 128;
        if (np1 > 0 && np2 > 0)
            if (dynamic_cast<ObjectVector*>(pv1) == nullptr) // don't need forces for pure particle halo
                CHOOSE_EXTERNAL(InteractionOut::NoAcc,   InteractionOut::NeedAcc, InteractionMode::Dilute, pair.handler() );
            else
                CHOOSE_EXTERNAL(InteractionOut::NeedAcc, InteractionOut::NeedAcc, InteractionMode::Dilute, pair.handler() );
    }

    PairwiseInteraction& getPairwiseInteraction(std::string pv1name, std::string pv2name)
    {
        auto it = intMap.find({pv1name, pv2name});
        if (it != intMap.end()) {
            debug("Using SPECIFIC parameters for PV pair '%s' -- '%s'", pv1name.c_str(), pv2name.c_str());
            return it->second;
        }
        else {
            debug("Using default parameters for PV pair '%s' -- '%s'", pv1name.c_str(), pv2name.c_str());
            return defaultPair;
        }
    }
};
