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
	int motionsOffset = 0;
	__forceinline__ __device__ void applyShift2extraData(char* packedExtra, float3 shift) const
	{
		LocalRigidObjectVector::RigidMotion motion;

		memcpy(&motion, packedExtra + motionsOffset, sizeof(motion));
		motion.r -= shift;
		memcpy(packedExtra + motionsOffset, &motion, sizeof(motion));
	}

	// TODO: implement this
	//float inertia[9];

	ROVview(RigidObjectVector* rov = nullptr, LocalRigidObjectVector* lrov = nullptr) :
		OVview(rov, lrov)
	{
		if (rov == nullptr || lrov == nullptr) return;

		motions = lrov->getDataPerObject<LocalRigidObjectVector::RigidMotion>("motions")->devPtr();

		// More fields
		J = rov->getInertiaTensor();
		J_1 = 1.0 / J;

		// Setup for the hack
		motionsOffset = 0;

		for (const auto& kv : lrov->getDataPerObjectMap())
		{
			if (kv.first == "motions") break;
			motionsOffset += kv.second->datatype_size();
		}
	}
};


