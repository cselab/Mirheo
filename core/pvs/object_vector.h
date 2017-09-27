#pragma once

#include <core/containers.h>
#include "particle_vector.h"

#include <core/logger.h>

struct ObjectMesh
{
	static const int maxDegree = 7;
	int nvertices, ntriangles;

	PinnedBuffer<int3> triangles;
	PinnedBuffer<int> adjacent, adjacent_second;
};

class LocalObjectVector: public LocalParticleVector
{
protected:
	int objSize  = 0;

public:

	PinnedBuffer<char*> extraDataPtrs;
	PinnedBuffer<int>   extraDataSizes;

	struct __align__(16) COMandExtent
	{
		float3 com, low, high;
	};

	int nObjects = 0;
	int packedObjSize_bytes = 0;
	DeviceBuffer<COMandExtent> comAndExtents;

	LocalObjectVector(const int objSize, const int nObjects = 0, cudaStream_t stream = 0) :
		LocalParticleVector(objSize*nObjects), objSize(objSize), nObjects(nObjects)
	{
		extraDataSizes.resize(1, stream, ResizeKind::resizeAnew);
		extraDataPtrs .resize(1, stream, ResizeKind::resizeAnew);

		extraDataSizes[0] = sizeof(COMandExtent);
		extraDataPtrs [0] = (char*)comAndExtents.devPtr();

		extraDataSizes.uploadToDevice(stream);
		extraDataPtrs .uploadToDevice(stream);

		resize(nObjects*objSize, stream, ResizeKind::resizeAnew);

		// Provide necessary alignment
		packedObjSize_bytes = ( (objSize*sizeof(Particle) + sizeof(COMandExtent) + sizeof(float4)-1) / sizeof(float4) ) * sizeof(float4);
	};


	virtual void resize(const int np, cudaStream_t stream, ResizeKind kind = ResizeKind::resizePreserve)
	{
		if (np % objSize != 0)
			die("Incorrect number of particles in object");

		nObjects = np / objSize;

		LocalParticleVector::resize(np, stream, kind);
		comAndExtents   .resize(nObjects, stream, kind);

		if ((char*)comAndExtents.devPtr() != extraDataPtrs[0])
		{
			extraDataPtrs[0] = (char*)comAndExtents.devPtr();
			extraDataPtrs.uploadToDevice(stream);
		}
	}

	virtual ~LocalObjectVector() = default;
};


class ObjectVector : public ParticleVector
{
protected:
	ObjectVector( std::string name, float mass, int objSize, LocalObjectVector *local, LocalObjectVector *halo ) :
		ParticleVector(name, mass, local, halo), objSize(objSize) {}

public:
	int objSize;
	ObjectMesh mesh;

	ObjectVector(std::string name, float mass, const int objSize, const int nObjects = 0) :
		ObjectVector( name, mass, objSize,
					  new LocalObjectVector(objSize, nObjects),
					  new LocalObjectVector(objSize, 0) )
	{}

	virtual void getMeshWithVertices(ObjectMesh* mesh, PinnedBuffer<Particle>* vertices, cudaStream_t stream);
	virtual void findExtentAndCOM(cudaStream_t stream);

	LocalObjectVector* local() { return static_cast<LocalObjectVector*>(_local); }
	LocalObjectVector* halo()  { return static_cast<LocalObjectVector*>(_halo);  }

	virtual ~ObjectVector() = default;
};


/**
 * GPU-compatibe struct of all the relevant data
 */
struct OVview : public PVview
{
	int nObjects, objSize;
	float objMass;

	LocalObjectVector::COMandExtent *comAndExtents;

	int extraDataNum;
	const int* extraDataSizes;  // extraDataNum integers
	char** extraData;           // extraDataNum arrays of extraDataSizes[i] bytes
	int packedObjSize_byte;     // total size of a packed object in bytes


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


	OVview(ObjectVector* ov, LocalObjectVector* lov) :
		PVview(static_cast<ParticleVector*>(ov), static_cast<LocalParticleVector*>(lov))
	{
		nObjects = lov->nObjects;
		objSize = ov->objSize;
		objMass = nObjects * mass;

		comAndExtents = lov->comAndExtents.devPtr();

		extraDataNum       = lov->extraDataSizes.size();
		extraDataSizes     = lov->extraDataSizes.devPtr();
		extraData          = lov->extraDataPtrs.devPtr();
		packedObjSize_byte = lov->packedObjSize_bytes;
	}
};






