#include "pairwise.h"

#include <core/cuda_common.h>
#include <core/celllist.h>
#include <core/pvs/particle_vector.h>
#include <core/logger.h>

#include "pairwise_kernels.h"

#include "pairwise_interactions/dpd.h"
#include "pairwise_interactions/lj.h"
#include "pairwise_interactions/lj_object_aware.h"

// Some not nice macro wrappers
// Cuda requires lambda defined in the same scope as where it is called...
#define DISPATCH_EXTERNAL(P1, P2, P3, TPP, INTERACTION_FUNCTION)                                                 \
do{ debug2("Dispatched to "#TPP" thread(s) per particle variant");                                               \
	computeExternalInteractions_##TPP##tpp<P1, P2, P3> <<< getNblocks(TPP*view.size, nth), nth, 0, stream >>>(   \
		view, cl2->cellInfo(), rc*rc, INTERACTION_FUNCTION); } while (0)

#define CHOOSE_EXTERNAL(P1, P2, P3, INTERACTION_FUNCTION)                                              \
do{  if (view.size < 1000  ) { DISPATCH_EXTERNAL(P1, P2, P3, 27, INTERACTION_FUNCTION); }              \
else if (view.size < 10000 ) { DISPATCH_EXTERNAL(P1, P2, P3, 9,  INTERACTION_FUNCTION); }              \
else if (view.size < 100000) { DISPATCH_EXTERNAL(P1, P2, P3, 3,  INTERACTION_FUNCTION); }              \
else                         { DISPATCH_EXTERNAL(P1, P2, P3, 1,  INTERACTION_FUNCTION); } } while(0)


template<class PariwiseInteraction>
void InteractionPair<PariwiseInteraction>::_compute(InteractionType type,
		ParticleVector* pv1, ParticleVector* pv2, CellList* cl1, CellList* cl2, const float t, cudaStream_t stream)
{
	static_assert(std::is_same<decltype(interaction.setup(pv1, pv2, cl1, cl2, t)), void>::value,
			"Pairwise interaction functor must provide member"
			"void setup(ParticleVector*, ParticleVector*, CellList*, CellList*, float)");

	auto& _inter = interaction;
	_inter.setup(pv1, pv2, cl1, cl2, t);

	auto core = [_inter] __device__ (
			const Particle dstP, const int dstId,
			const Particle srcP, const int srcId ) {

		static_assert(std::is_same<decltype(_inter.operator()(dstP, dstId, srcP, srcId)), float3>::value,
					"Pairwise interaction functor must provide member"
					"__device__ float3 operator()(Particle, Particle, int, int)");

		return _inter(dstP, dstId, srcP, srcId);
	};

	if (type == InteractionType::Regular)
	{
		/*  Self interaction */
		if (pv1 == pv2)
		{
			const int np = pv1->local()->size();
			debug("Computing internal forces for %s (%d particles)", pv1->name.c_str(), np);

			const int nth = 128;
//			if (np > 0)
//				computeSelfInteractions<<< getNblocks(np, nth), nth, 0, stream >>>(np, cl1->cellInfo(), rc*rc, core);
			auto cinfo = cl1->cellInfo();
			if (np > 0)
				computeSelfInteractions <<<getNblocks(np, nth), nth, 0, stream>>> (
						np, cinfo, rc*rc, core);
		}
		else /*  External interaction */
		{
			const int np1 = pv1->local()->size();
			const int np2 = pv2->local()->size();
			debug("Computing external forces for %s - %s (%d - %d particles)", pv1->name.c_str(), pv2->name.c_str(), np1, np2);

			auto view = create_PVview(pv1, pv1->local());
			const int nth = 128;
			if (np1 > 0 && np2 > 0)
				CHOOSE_EXTERNAL(true, true, true, core);
		}
	}

	/*  Halo interaction */
	if (type == InteractionType::Halo)
	{
		const int np1 = pv1->halo()->size();  // note halo here
		const int np2 = pv2->local()->size();
		debug("Computing halo forces for %s(halo) - %s (%d - %d particles)", pv1->name.c_str(), pv2->name.c_str(), np1, np2);

		auto view = create_PVview(pv1, pv1->halo());
		const int nth = 128;
		if (np1 > 0 && np2 > 0)
			if (dynamic_cast<ObjectVector*>(pv1) == nullptr) // don't need forces for pure particle halo
				CHOOSE_EXTERNAL(false, true, false, core);
			else
				CHOOSE_EXTERNAL(true,  true, false, core);
	}
}

template class InteractionPair<Pairwise_DPD>;
template class InteractionPair<Pairwise_LJ>;
template class InteractionPair<Pairwise_LJObjectAware>;


