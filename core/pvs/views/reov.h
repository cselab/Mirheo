#pragma once

/**
 * GPU-compatible struct of all the relevant data
 */
struct REOVview : public ROVview
{
	float3 axes    = {0,0,0};
	float3 invAxes = {0,0,0};

	REOVview(RigidEllipsoidObjectVector* reov = nullptr, LocalObjectVector* lov = nullptr) :
		ROVview(reov, lov)
	{
		if (reov == nullptr || lov == nullptr) return;

		// More fields
		axes = reov->axes;
		invAxes = 1.0 / axes;
	}
};

struct REOVviewWithOldMotion : public REOVview
{
	RigidMotion *old_motions = nullptr;

	REOVviewWithOldMotion(RigidEllipsoidObjectVector* reov = nullptr, LocalObjectVector* lov = nullptr) :
		REOVview(reov, lov)
	{
		if (reov == nullptr || lov == nullptr) return;

		old_motions = lov->extraPerObject.getData<RigidMotion>("old_motions")->devPtr();
	}
};
