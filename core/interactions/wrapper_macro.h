#pragma once

#define WRAP_INTERACTON(INTERACTION_FUNCTION)                                                                                                                         \
if (type == InteractionType::Regular)                                                                                                                             \
{                                                                                                                                                                 \
	/*  Self interaction */                                                                                                                                       \
	if (pv1 == pv2)                                                                                                                                               \
	{                                                                                                                                                             \
		debug2("Computing internal forces for %s (%d particles)", pv1->name.c_str(), pv1->local()->size());                                                       \
																																								  \
		const int nth = 128;                                                                                                                                      \
		if (pv1->local()->size() > 0)                                                                                                                             \
			computeSelfInteractions<<< (pv1->local()->size() + nth - 1) / nth, nth, 0, stream >>>(                                                                \
					pv1->local()->size(), (float4*)cl->coosvels->devPtr(), (float*)cl->forces->devPtr(),                                                          \
					cl->cellInfo(), cl->cellsStartSize.devPtr(), rc*rc, INTERACTION_FUNCTION);                                                                    \
	}                                                                                                                                                             \
	else /*  External interaction */                                                                                                                              \
	{                                                                                                                                                             \
		debug2("Computing external forces for %s - %s (%d - %d particles)", pv1->name.c_str(), pv2->name.c_str(), pv1->local()->size(), pv2->local()->size());    \
		if (pv1->local()->size() < pv2->local()->size())                                                                                                          \
			std::swap(pv1, pv2);                                                                                                                                  \
			                                                                                                                                                      \
		const int nth = 128;                                                                                                                                      \
		if (pv1->local()->size() > 0 && pv2->local()->size() > 0)                                                                                                 \
			computeExternalInteractions<true, true, true> <<< (pv2->local()->size() + nth - 1) / nth, nth, 0, stream >>>(                                         \
					pv2->local()->size(),                                                                                                                         \
					(float4*)pv2->local()->coosvels.devPtr(), (float*)pv2->local()->forces.devPtr(),                                                              \
					(float4*)cl->coosvels->devPtr(), (float*)cl->forces->devPtr(),                                                                                \
					cl->cellInfo(), cl->cellsStartSize.devPtr(),                                                                                                  \
					rc*rc, INTERACTION_FUNCTION);                                                                                                                 \
	}                                                                                                                                                             \
}                                                                                                                                                                 \
																																								  \
/*  Halo interaction */                                                                                                                                           \
if (type == InteractionType::Halo)                                                                                                                                \
{                                                                                                                                                                 \
	debug2("Computing halo forces for %s - %s(halo) (%d - %d particles)", pv1->name.c_str(), pv2->name.c_str(), pv1->local()->size(), pv2->halo()->size());       \
																																								  \
	const int nth = 128;                                                                                                                                          \
	if (pv1->local()->size() > 0 && pv2->halo()->size() > 0)                                                                                                      \
		computeExternalInteractions<false, true, false> <<< (pv2->halo()->size() + nth - 1) / nth, nth, 0, stream >>>(                                            \
				pv2->halo()->size(),                                                                                                                              \
				(float4*)pv2->halo()->coosvels.devPtr(), nullptr,                                                                                                 \
				(float4*)cl->coosvels->devPtr(), (float*)cl->forces->devPtr(),                                                                                    \
				cl->cellInfo(), cl->cellsStartSize.devPtr(),                                                                                                      \
				rc*rc, INTERACTION_FUNCTION);                                                                                                                     \
}
