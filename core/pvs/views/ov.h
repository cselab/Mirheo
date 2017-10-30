#pragma once

#include <core/rigid_kernels/rigid_motion.h>

/**
 * GPU-compatible struct of all the relevant data
 */
struct OVview : public PVview
{
	int nObjects = 0, objSize = 0;
	float objMass = 0, invObjMass = 0;

	LocalObjectVector::COMandExtent *comAndExtents = nullptr;
	int* ids = nullptr;

	OVview(ObjectVector* ov = nullptr, LocalObjectVector* lov = nullptr) :
		PVview(ov, lov)
	{
		if (ov == nullptr || lov == nullptr) return;

		// More fields
		nObjects = lov->nObjects;
		objSize = ov->objSize;
		objMass = objSize * mass;
		invObjMass = 1.0 / objMass;

		// Required data per object
		comAndExtents = lov->getDataPerObject<LocalObjectVector::COMandExtent>("com_extents")->devPtr();
		ids           = lov->getDataPerObject<int>("ids")->devPtr();
	}
};

struct OVviewWithAreaVolume : public OVview
{
	float2* area_volumes = nullptr;

	OVviewWithAreaVolume(ObjectVector* ov = nullptr, LocalObjectVector* lov = nullptr) :
		OVview(ov, lov)
	{
		if (ov == nullptr || lov == nullptr) return;

		area_volumes = lov->getDataPerObject<float2>("area_volumes")->devPtr();
	}
};

struct OVviewWithOldPartilces : public OVview
{
	float4* old_particles = nullptr;

	OVviewWithOldPartilces(ObjectVector* ov = nullptr, LocalObjectVector* lov = nullptr) :
		OVview(ov, lov)
	{
		if (ov == nullptr || lov == nullptr) return;

		old_particles = reinterpret_cast<float4*>( lov->getDataPerParticle<Particle>("old_particles")->devPtr() );
	}
};

// Names of data fields that have to be shifted when redistributed
const static std::map<std::string, int> dataToShift{ {"motions", 0}, {"old_motions", 0} };

struct OVviewWithExtraData : public OVview
{
	int extraDataNum = 0;
	const int* extraDataSizes = nullptr; // extraDataNum integers
	char** extraData = nullptr;          // extraDataNum arrays of extraDataSizes[i] bytes
	int packedObjSize_byte = 0;          // total size of a packed object in bytes

	int  nToShift = 0;
	int* shiftOffsets = nullptr;

	__forceinline__ __device__ void applyShift2extraData(char* packedExtra, float3 shift) const
	{
		for (int i=0; i<nToShift; i++)
		{
			RigidReal3 v;
			memcpy(&v, packedExtra + shiftOffsets[i], sizeof(RigidReal3));
			v.x -= shift.x;
			v.y -= shift.y;
			v.z -= shift.z;
			memcpy(packedExtra + shiftOffsets[i], &v, sizeof(RigidReal3));
		}
	}

	// TODO: can be improved with binsearch of ptr per warp
	__forceinline__ __device__ void packExtraData(int srcObjId, char* destination) const
	{
		for (int ptrId = 0; ptrId < extraDataNum; ptrId++)
		{
			const int size = extraDataSizes[ptrId];
			for (int i = threadIdx.x; i < size; i += blockDim.x)
				destination[i] = extraData[ptrId][srcObjId*size + i];

			destination += size;
		}
	}

	__forceinline__ __device__ void unpackExtraData(int dstObjId, const char* source) const
	{
		for (int ptrId = 0; ptrId < extraDataNum; ptrId++)
		{
			const int size = extraDataSizes[ptrId];
			for (int i = threadIdx.x; i < size; i += blockDim.x)
				extraData[ptrId][dstObjId*size + i] = source[i];

			source += size;
		}
	}

	OVviewWithExtraData(ObjectVector* ov = nullptr, LocalObjectVector* lov = nullptr, cudaStream_t stream = 0) :
		OVview(ov, lov)
	{
		if (ov == nullptr || lov == nullptr) return;

		// Extra data per object
		extraDataNum = lov->getDataPerObjectMap().size();

		lov->extraDataPtrs. resize_anew(extraDataNum);
		lov->extraDataSizes.resize_anew(extraDataNum);

		int n = 0;
		bool upload = false;

		packedObjSize_byte = objSize * sizeof(Particle);
		for (const auto& kv : lov->getDataPerObjectMap())
		{
			lov->extraDataSizes[n] = kv.second->datatype_size();
			packedObjSize_byte += lov->extraDataSizes[n];

			void* ptr = kv.second->genericDevPtr();
			if (ptr != lov->extraDataPtrs[n]) upload = true;
			lov->extraDataPtrs[n] = reinterpret_cast<char*>(ptr);

			n++;
		}

		// Align packed size to float4 size
		packedObjSize_byte = ( (packedObjSize_byte + sizeof(float4) - 1) / sizeof(float4) ) * sizeof(float4);

		if (upload)
		{
			lov->extraDataPtrs. uploadToDevice(stream);
			lov->extraDataSizes.uploadToDevice(stream);
		}

		extraDataSizes = lov->extraDataSizes.devPtr();
		extraData      = lov->extraDataPtrs. devPtr();

		// Setup offsets for the data needing shifts
		int offset = 0;
		nToShift = 0;
		upload = false;

		for (auto& kv : lov->getDataPerObjectMap())
		{
			auto it = dataToShift.find(kv.first);
			if (it != dataToShift.end())
			{
				lov->shiftingDataOffsets.resize_anew(nToShift + 1);
				int curOffset = offset + it->second;
				if (lov->shiftingDataOffsets[nToShift] != curOffset) upload = true;
				lov->shiftingDataOffsets[nToShift] = curOffset;

				nToShift++;
			}

			offset += kv.second->datatype_size();
		}

		if (upload) lov->shiftingDataOffsets.uploadToDevice(stream);
		shiftOffsets = lov->shiftingDataOffsets.devPtr();
	}
};

