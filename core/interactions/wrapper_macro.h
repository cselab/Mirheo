#pragma once

// TODO: change to particle and cell-list views

#define WRAP_INTERACTON(INTERACTION_FUNCTION)                                                                                 \
if (type == InteractionType::Regular)                                                                                         \
{                                                                                                                             \
	/*  Self interaction */                                                                                                   \
	if (pv1 == pv2)                                                                                                           \
	{                                                                                                                         \
		debug2("Computing internal forces for %s (%d particles)", pv1->name.c_str(), pv1->local()->size());                   \
                                                                                                                              \
		const int nth = 128;                                                                                                  \
		if (pv1->local()->size() > 0)                                                                                         \
			computeSelfInteractions<<< (pv1->local()->size() + nth - 1) / nth, nth, 0, stream >>>(                            \
					pv1->local()->size(), (float4*)cl1->coosvels->devPtr(), (float*)cl1->forces->devPtr(),                    \
					cl1->cellInfo(), cl1->cellsStartSize.devPtr(), rc*rc, INTERACTION_FUNCTION);                              \
	}                                                                                                                         \
	else /*  External interaction */                                                                                          \
	{                                                                                                                         \
		debug2("Computing external forces for %s - %s (%d - %d particles)",                                                   \
				pv1->name.c_str(), pv2->name.c_str(), pv1->local()->size(), pv2->local()->size());                            \
                                                                                                                              \
		const int nth = 128;                                                                                                  \
		if (pv1->local()->size() > 0 && pv2->local()->size() > 0)                                                             \
			computeExternalInteractions<true, true, true> <<< (pv1->local()->size() + nth - 1) / nth, nth, 0, stream >>>(     \
					pv1->local()->size(),                                                                                     \
					(float4*)pv1->local()->coosvels.devPtr(), (float*)pv1->local()->forces.devPtr(),                          \
					(float4*)cl2->coosvels->devPtr(), (float*)cl2->forces->devPtr(),                                          \
					cl2->cellInfo(), cl2->cellsStartSize.devPtr(),                                                            \
					rc*rc, INTERACTION_FUNCTION);                                                                             \
	}                                                                                                                         \
}                                                                                                                             \
                                                                                                                              \
/*  Halo interaction */                                                                                                       \
if (type == InteractionType::Halo)                                                                                            \
{                                                                                                                             \
	debug2("Computing halo forces for %s(halo) - %s (%d - %d particles)",                                                     \
			pv1->name.c_str(), pv2->name.c_str(), pv1->halo()->size(), pv2->local()->size());                                 \
                                                                                                                              \
	const int nth = 128;                                                                                                      \
	if (pv1->halo()->size() > 0 && pv2->local()->size() > 0)                                                                  \
		if (dynamic_cast<ObjectVector*>(pv1) == nullptr) /* don't need acceleartions for pure particle halo */                \
			computeExternalInteractions<false, true, false>  <<< (pv1->halo()->size() + nth - 1) / nth, nth, 0, stream >>>(   \
					pv1->halo()->size(),                                                                                      \
					(float4*)pv1->halo()->coosvels.devPtr(), (float*)pv1->halo()->forces.devPtr(),                            \
					(float4*)cl2->coosvels->devPtr(), (float*)cl2->forces->devPtr(),                                          \
					cl2->cellInfo(), cl2->cellsStartSize.devPtr(),                                                            \
					rc*rc, INTERACTION_FUNCTION);                                                                             \
		else                                                                                                                  \
			computeExternalInteractions<true, true, false> <<< (pv1->halo()->size() + nth - 1) / nth, nth, 0, stream >>>(     \
					pv1->halo()->size(),                                                                                      \
					(float4*)pv1->halo()->coosvels.devPtr(), (float*)pv1->halo()->forces.devPtr(),                            \
					(float4*)cl2->coosvels->devPtr(), (float*)cl2->forces->devPtr(),                                          \
					cl2->cellInfo(), cl2->cellsStartSize.devPtr(),                                                            \
					rc*rc, INTERACTION_FUNCTION);                                                                             \
}

