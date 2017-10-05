#pragma once

/**
 * GPU-compatible struct of all the relevant data
 */
struct ROVview : public OVview
{
	LocalRigidObjectVector::RigidMotion *motions = nullptr;

	float3 J   = {0,0,0};
	float3 J_1 = {0,0,0};

	/**
	 * FIXME: this is a hack.
	 */
	int motionsOffset;
	__forceinline__ __device__ void applyShift2extraData(char* packedExtra, float3 shift) const
	{
		LocalRigidObjectVector::RigidMotion motion;

		memcpy(&motion, packedExtra + motionsOffset, sizeof(motion));
		motion.r -= shift;
		memcpy(packedExtra + motionsOffset, &motion, sizeof(motion));
	}

	// TODO: implement this
	//float inertia[9];
};


static ROVview create_ROVview(RigidObjectVector* rov, LocalRigidObjectVector* lrov)
{
	// Create a default view
	ROVview view;
	if (rov == nullptr || lrov == nullptr)
		return view;

	view.motions = lrov->getDataPerObject<LocalRigidObjectVector::RigidMotion>("motions")->devPtr();

	view.OVview::operator= ( create_OVview(rov, lrov) );

	// More fields
	view.J = rov->getInertiaTensor();
	view.J_1 = 1.0 / view.J;

	// Setup for the hack
	view.motionsOffset = 0;

	for (const auto& kv : lrov->getDataPerObjectMap())
	{
		if (kv.first == "motions") break;
		view.motionsOffset += kv.second->datatype_size();
	}

	return view;
}
