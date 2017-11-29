#pragma once

/**
 * GPU-compatible struct of all the relevant data
 */
struct ROVview : public OVview
{
	RigidMotion *motions = nullptr;

	float3 J   = {0,0,0};
	float3 J_1 = {0,0,0};

	// TODO: implement this
	//float inertia[9];

	ROVview(RigidObjectVector* rov = nullptr, LocalObjectVector* lov = nullptr) :
		OVview(rov, lov)
	{
		if (rov == nullptr || lov == nullptr) return;

		motions = lov->extraPerObject.getData<RigidMotion>("motions")->devPtr();

		// More fields
		J = rov->J;
		J_1 = 1.0 / J;
	}
};

struct ROVviewWithOldMotion : public ROVview
{
	RigidMotion *old_motions = nullptr;

	ROVviewWithOldMotion(RigidObjectVector* rov = nullptr, LocalObjectVector* lov = nullptr) :
		ROVview(rov, lov)
	{
		if (rov == nullptr || lov == nullptr) return;

		old_motions = lov->extraPerObject.getData<RigidMotion>("old_motions")->devPtr();
	}
};

