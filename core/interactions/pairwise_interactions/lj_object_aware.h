#pragma once

#include <core/datatypes.h>
#include <core/pvs/particle_vector.h>
#include <core/pvs/object_vector.h>

class Pairwise_LJObjectAware
{
public:
	Pairwise_LJObjectAware(float rc, float sigma, float epsilon) :
		lj(rc, sigma, epsilon)
	{	}

	void setup(ParticleVector* pv1, ParticleVector* pv2, CellList* cl1, CellList* cl2, float t)
	{
		auto ov1 = dynamic_cast<ObjectVector*>(pv1);
		auto ov2 = dynamic_cast<ObjectVector*>(pv2);
		if (ov1 == nullptr && ov2 == nullptr)
			die("Object-aware LJ interaction can only be used with objects");

		self = (pv1 == pv2);

		view1 = OVview(ov1, ov1 ? ov1->local() : nullptr);
		view2 = OVview(ov2, ov2 ? ov2->local() : nullptr);

		if (view1.comAndExtents == nullptr && view2.comAndExtents == nullptr)
			warn("Neither of the pvs (%s or %s) has required property 'com_extents', trying to move on regardless",
					pv1->name.c_str(), pv2->name.c_str());
	}

	__device__ inline float3 operator()(Particle dst, int dstId, Particle src, int srcId) const
	{
		//    _____             _____
		//  /       \    dr   /       \
		// |   DST  *| <---- |*  SRC   |
		//  \ ______/         \ ______/
		//     <---             --->
		//       f               -f
		// Forces must only repel DST from SRC
		//
		const int dstObjId = dstId % view1.objSize;
		const int srcObjId = srcId % view2.objSize;

		if (dstObjId == srcObjId && self) return make_float3(0.0f);

		float3 dstCom = make_float3(0.0f);
		float3 srcCom = make_float3(0.0f);

		const bool isDstObj = view1.comAndExtents != nullptr;
		const bool isSrcObj = view2.comAndExtents != nullptr;

		if (isDstObj) dstCom = view1.comAndExtents[dstObjId].com;
		if (isSrcObj) srcCom = view2.comAndExtents[srcObjId].com;

		float3 f = lj(dst, dstId, src, srcId);

		if ( isDstObj && dot(f,  dstCom - dst.r) < 0.0f ) f = make_float3(0.0f);
		if ( isSrcObj && dot(-f, srcCom - src.r) < 0.0f ) f = make_float3(0.0f);

		return f;
	}


private:

	bool self;
	OVview view1, view2;

	Pairwise_LJ lj;
};
