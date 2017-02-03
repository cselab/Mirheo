#pragma once

class ParticleVector;
class CellList;

//==================================================================================================================
// DPD interactions
//==================================================================================================================

void interactionDPDSelf (ParticleVector* pv, CellList* cl, const float t, cudaStream_t stream,
		float adpd, float gammadpd, float sigma_dt, float rc);

void interactionDPDHalo (ParticleVector* pv1, ParticleVector* pv2, CellList* cl, const float t, cudaStream_t stream,
		float adpd, float gammadpd, float sigma_dt, float rc);

void interactionDPDExternal (ParticleVector* pv1, ParticleVector* pv2, CellList* cl, const float t, cudaStream_t stream,
		float adpd, float gammadpd, float sigma_dt, float rc);
