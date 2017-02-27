#pragma once

class ParticleVector;
class CellList;

//==================================================================================================================
// DPD interactions
//==================================================================================================================

enum InteractionType { Regular, Halo };

void interactionDPD (InteractionType type, ParticleVector* pv1, ParticleVector* pv2, CellList* cl, const float t, cudaStream_t stream,
		float adpd, float gammadpd, float sigma_dt, float rc);
