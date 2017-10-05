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
};

static OVview create_OVview(ObjectVector* ov, LocalObjectVector* lov)
{
	// Create a default view
	OVview view;
	if (ov == nullptr || lov == nullptr)
		return view;

	view.PVview::operator= ( create_PVview(ov, lov) );

	// More fields
	view.nObjects = lov->nObjects;
	view.objSize = ov->objSize;
	view.objMass = view.objSize * view.mass;
	view.invObjMass = 1.0 / view.objMass;

	// Required data per object
	view.comAndExtents = lov->getDataPerObject<LocalObjectVector::COMandExtent>("com_extents")->devPtr();
	view.ids           = lov->getDataPerObject<int>("ids")->devPtr();

	return view;
}

static OVviewWithExtraData create_OVviewWithExtraData(ObjectVector* ov, LocalObjectVector* lov, cudaStream_t stream)
{
	// Create a default view
	OVviewWithExtraData view;
	if (ov == nullptr || lov == nullptr)
		return view;

	view.OVview::operator= ( create_OVview(ov, lov) );

	// Extra data per object
	view.extraDataNum = lov->getDataPerObjectMap().size();

	lov->extraDataPtrs. resize_anew(view.extraDataNum);
	lov->extraDataSizes.resize_anew(view.extraDataNum);

	int n = 0;
	bool upload = false;

	view.packedObjSize_byte = view.objSize * sizeof(Particle);
	for (const auto& kv : lov->getDataPerObjectMap())
	{
		lov->extraDataSizes[n] = kv.second->datatype_size();
		view.packedObjSize_byte += lov->extraDataSizes[n];

		void* ptr = kv.second->genericDevPtr();
		if (ptr != lov->extraDataPtrs[n]) upload = true;
		lov->extraDataPtrs[n] = reinterpret_cast<char*>(ptr);

		n++;
	}

	// Align packed size to float4 size
	view.packedObjSize_byte = ( (view.packedObjSize_byte + sizeof(float4) - 1) / sizeof(float4) ) * sizeof(float4);


	if (upload)
	{
		lov->extraDataPtrs.uploadToDevice(stream);
		lov->extraDataSizes.uploadToDevice(stream);
	}

	view.extraDataSizes     = lov->extraDataSizes.devPtr();
	view.extraData          = lov->extraDataPtrs.devPtr();

	return view;
}
