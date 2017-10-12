#pragma once

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

struct OVviewWithExtraData : public OVview
{
	int extraDataNum = 0;
	const int* extraDataSizes = nullptr; // extraDataNum integers
	char** extraData = nullptr;          // extraDataNum arrays of extraDataSizes[i] bytes
	int packedObjSize_byte = 0;          // total size of a packed object in bytes


	// TODO: can be improved with binsearch of ptr per warp
	__forceinline__ __device__ void packExtraData(int srcObjId, char* destanation) const
	{
		int baseId = 0;

		for (int ptrId = 0; ptrId < extraDataNum; ptrId++)
		{
			const int size = extraDataSizes[ptrId];

			for (int i = threadIdx.x; i < size; i += blockDim.x)
				destanation[baseId+i] = extraData[ptrId][srcObjId*size + i];

			baseId += extraDataSizes[ptrId];
		}
	}

	__forceinline__ __device__ void unpackExtraData(int dstObjId, const char* source) const
	{
		int baseId = 0;

		for (int ptrId = 0; ptrId < extraDataNum; ptrId++)
		{
			const int size = extraDataSizes[ptrId];
			for (int i = threadIdx.x; i < size; i += blockDim.x)
				extraData[ptrId][dstObjId*size + i] = source[baseId+i];

			baseId += extraDataSizes[ptrId];
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
	}
};

