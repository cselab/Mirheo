#pragma once

/**
 * GPU-compatible struct of all the relevant data
 */
struct REOVview : public ROVview
{
	float3 axes    = {0,0,0};
	float3 invAxes = {0,0,0};

	REOVview(RigidEllipsoidObjectVector* reov = nullptr, LocalRigidEllipsoidObjectVector* lreov = nullptr) :
		ROVview(reov, lreov)
	{
		if (reov == nullptr || lreov == nullptr) return;

		// More fields
		axes = reov->axes;
		invAxes = 1.0 / axes;
	}
};

