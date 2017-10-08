#pragma once

// TODO: change to particle and cell-list views

#define DISPATCH_EXTERNAL(P1, P2, P3, TPP, INTERACTION_FUNCTION)                                                 \
do{ debug2("Dispatched to "#TPP" thread(s) per particle variant");                                               \
	computeExternalInteractions_##TPP##tpp<P1, P2, P3> <<< getNblocks(TPP*view.size, nth), nth, 0, stream >>>(   \
		view, cl2->cellInfo(), rc*rc, INTERACTION_FUNCTION); } while (0)

#define CHOOSE_EXTERNAL(P1, P2, P3, INTERACTION_FUNCTION)                                             \
do{  if (view.size < 5000 ) { DISPATCH_EXTERNAL(P1, P2, P3, 27, INTERACTION_FUNCTION); }              \
else if (view.size < 10000) { DISPATCH_EXTERNAL(P1, P2, P3, 9,  INTERACTION_FUNCTION); }              \
else if (view.size < 30000) { DISPATCH_EXTERNAL(P1, P2, P3, 3,  INTERACTION_FUNCTION); }              \
else                        { DISPATCH_EXTERNAL(P1, P2, P3, 1,  INTERACTION_FUNCTION); } } while(0)

#define WRAP_INTERACTON(INTERACTION_FUNCTION)                                                                       \
if (type == InteractionType::Regular)                                                                               \
{                                                                                                                   \
	/*  Self interaction */                                                                                         \
	if (pv1 == pv2)                                                                                                 \
	{                                                                                                               \
		const int np = pv1->local()->size();                                                                        \
		debug("Computing internal forces for %s (%d particles)", pv1->name.c_str(), np);                            \
                                                                                                                    \
		const int nth = 128;                                                                                        \
		if (np > 0)                                                                                                 \
			computeSelfInteractions<<< getNblocks(np, nth), nth, 0, stream >>>(                                     \
					np, cl1->cellInfo(), rc*rc, INTERACTION_FUNCTION);                                              \
	}                                                                                                               \
	else /*  External interaction */                                                                                \
	{                                                                                                               \
		const int np1 = pv1->local()->size();                                                                       \
		const int np2 = pv2->local()->size();                                                                       \
		debug("Computing external forces for %s - %s (%d - %d particles)",                                          \
				pv1->name.c_str(), pv2->name.c_str(), np1, np2);                                                    \
                                                                                                                    \
		auto view = create_PVview(pv1, pv1->local());                                                               \
		const int nth = 128;                                                                                        \
		if (np1 > 0 && np2 > 0)                                                                                     \
			CHOOSE_EXTERNAL(true, true, true, INTERACTION_FUNCTION);                                                \
	}                                                                                                               \
}                                                                                                                   \
                                                                                                                    \
/*  Halo interaction */                                                                                             \
if (type == InteractionType::Halo)                                                                                  \
{                                                                                                                   \
	const int np1 = pv1->halo()->size();  /* note halo here */                                                      \
	const int np2 = pv2->local()->size();                                                                           \
	debug("Computing halo forces for %s(halo) - %s (%d - %d particles)",                                            \
			pv1->name.c_str(), pv2->name.c_str(), np1, np2);                                                        \
                                                                                                                    \
	auto view = create_PVview(pv1, pv1->halo());                                                                    \
	const int nth = 128;                                                                                            \
	if (np1 > 0 && np2 > 0)                                                                                         \
		if (dynamic_cast<ObjectVector*>(pv1) == nullptr) /* don't need acceleartions for pure particle halo */      \
			CHOOSE_EXTERNAL(false, true, false, INTERACTION_FUNCTION);                                              \
		else                                                                                                        \
			CHOOSE_EXTERNAL(true,  true, false, INTERACTION_FUNCTION);                                              \
}

