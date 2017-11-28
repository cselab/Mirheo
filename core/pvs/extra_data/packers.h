#pragma once

#include <cassert>
#include "../particle_vector.h"
#include "../object_vector.h"


/**
 * Class that packs nChannels of arbitrary data into a chunk of contiguous memory
 * or unpacks it in the same manner
 */
struct DevicePacker
{
	int packedSize_byte = 0;

	int nChannels = 0;                    /// number of data channels to pack / unpack
	int* channelSizes        = nullptr;   /// size if bytes of each channel entry, e.g. sizeof(Particle)
	int* channelShiftOffsets = nullptr;   /** if offset is >= 0, then shift will be applied to the channel when packing
	 *  3 real numbers will be changed (+=) by shift at the offset from the structure start
	 *  E.g. struct Smth { float4 dummy; float3 needShift; }  =>  offset is sizeof(float4)
	 *  Note that padding may be applied by the compiler
	 */

	int* channelShiftTypes   = nullptr;   /// if type is 4, then treat data to shift as float3, if it is 8 -- as double3
	char** channelData       = nullptr;   /// device pointers of the packed data

	/**
	 * Pack entity with id srcId into memory starting with dstAddr
	 * Don't apply no shifts
	 */
	__forceinline__ __device__ void pack(int srcId, char* dstAddr) const
	{
		_packShift<false> (srcId, dstAddr, make_float3(0));
	}

	/**
	 * Pack entity with id srcId into memory starting with dstAddr
	 * Apply shifts where needed
	 */
	__forceinline__ __device__ void packShift(int srcId, char* dstAddr, float3 shift) const
	{
		_packShift<true>  (srcId, dstAddr, shift);
	}

	/**
	 * Unpack entity from memory by srcAddr to the channels to id dstId
	 */
	__forceinline__ __device__ void unpack(const char* srcAddr, int dstId) const
	{
		for (int i = 0; i < nChannels; i++)
		{
			copy(channelData[i] + channelSizes[i]*dstId, srcAddr, channelSizes[i]);
			srcAddr += channelSizes[i];
		}
	}

private:

	/**
	 * Copy nchunks*sizeof(T) bytes from from to to
	 */
	template<typename T>
	__forceinline__ __device__ void _copy(char* to, const char* from, int nchunks) const
	{
		auto _to   = (T*)to;
		auto _from = (const T*)from;

#pragma unroll 2
		for (int i=0; i<nchunks; i++)
			_to[i] = _from[i];
	}

	/**
	 * Copy size_bytes bytes from from to to
	 * Speed up copying by choosing the widest possible data type
	 * and calling the appropriate _copy function
	 */
	__forceinline__ __device__ void copy(char* to, const char* from, int size_bytes) const
	{
		assert(size_bytes % 4 == 0);

		if (size_bytes % 16 == 0)
			_copy<int4>(to, from, size_bytes / 16);
		else if (size_bytes % 8 == 0)
			_copy<int2>(to, from, size_bytes / 8);
		else
			_copy<int> (to, from, size_bytes / 4);
	}


	/**
	 * Packing implementation
	 * Template parameter NEEDSHIFT governs shifting
	 */
	template <bool NEEDSHIFT>
	__forceinline__ __device__ void _packShift(int srcId, char* dstAddr, float3 shift) const
	{
		for (int i = 0; i < nChannels; i++)
		{
			copy(dstAddr, channelData[i] + channelSizes[i]*srcId, channelSizes[i]);

//			if (NEEDSHIFT && channelShiftOffsets[i] >= 0)
//			{
//				char* shAddr = dstAddr + channelShiftOffsets[i];
//
//				if (channelShiftTypes[i] == 4)
//				{
//					float* fShAddr = (float*) shAddr;
//					fShAddr[0] += shift.x;
//					fShAddr[1] += shift.y;
//					fShAddr[2] += shift.z;
//				}
//				else if (channelShiftTypes[i] == 8)
//				{
//					double* dShAddr = (double*) shAddr;
//					dShAddr[0] += shift.x;
//					dShAddr[1] += shift.y;
//					dShAddr[2] += shift.z;
//				}
//				// else KILLALLHUMANSNOW!
//			}

			dstAddr += channelSizes[i];
		}
	}
};


/**
 * Class that uses DevicePacker to pack a single particle entity
 */
struct ParticlePacker : public DevicePacker
{
	ParticlePacker(ParticleVector* pv = nullptr, LocalParticleVector* lpv = nullptr, cudaStream_t stream = 0)
	{
		if (pv == nullptr || lpv == nullptr) return;

		auto& manager = lpv->extraPerParticle;

		// 1 channel for particles themselves + extra data per particle from manager
		nChannels = manager.getDataMap().size() + 1;

		manager.channelPtrs.        resize_anew(nChannels);
		manager.channelSizes.       resize_anew(nChannels);
		manager.channelShiftOffsets.resize_anew(nChannels);
		manager.channelShiftTypes.  resize_anew(nChannels);

		int n = 0;
		bool upload = false;

		auto registerChannel = [&] (int sz, char* ptr, int offset, int typesize) {

			if (ptr != manager.channelPtrs[n]) upload = true;

			manager.channelSizes[n] = sz;
			manager.channelPtrs[n] = ptr;
			manager.channelShiftOffsets[n] = offset;
			manager.channelShiftTypes[n] = typesize;

			packedSize_byte += sz;
			n++;
		};

		registerChannel(
				sizeof(Particle),
				reinterpret_cast<char*>(lpv->coosvels.devPtr()),
				0,
				sizeof(float) );


		for (const auto& kv : manager.getDataMap())
		{
			if (manager.needExchange(kv.first))
			{
				int sz = kv.second->datatype_size();

				if (sz % sizeof(int) != 0)
					die("Size of extra data per particle should be divisible by 4 bytes (PV '%s', data entry '%s')",
							pv->name.c_str(), kv.first.c_str());

				registerChannel(
						sz,
						reinterpret_cast<char*>(kv.second->genericDevPtr()),
						manager.shiftOffsetType(kv.first).first,
						manager.shiftOffsetType(kv.first).second  );
			}
		}

		if (upload)
		{
			manager.channelPtrs.        uploadToDevice(stream);
			manager.channelSizes.       uploadToDevice(stream);
			manager.channelShiftOffsets.uploadToDevice(stream);
			manager.channelShiftTypes.  uploadToDevice(stream);
		}

		channelData         = manager.channelPtrs.        devPtr();
		channelSizes        = manager.channelSizes.       devPtr();
		channelShiftOffsets = manager.channelShiftOffsets.devPtr();
		channelShiftTypes   = manager.channelShiftTypes.  devPtr();
	}
};


/**
 * Class that uses DevicePacker to pack extra data per object
 */
struct ObjectExtraPacker : public DevicePacker
{
	ObjectExtraPacker(ObjectVector* ov = nullptr, LocalObjectVector* lov = nullptr, cudaStream_t stream = 0)
	{
		if (ov == nullptr || lov == nullptr) return;

		auto& manager = lov->extraPerObject;

		nChannels = manager.getDataMap().size();

		manager.channelPtrs.        resize_anew(nChannels);
		manager.channelSizes.       resize_anew(nChannels);
		manager.channelShiftOffsets.resize_anew(nChannels);
		manager.channelShiftTypes.  resize_anew(nChannels);

		int n = 0;
		bool upload = false;

		auto registerChannel = [&] (int sz, char* ptr, int offset, int typesize) {

			if (ptr != manager.channelPtrs[n]) upload = true;

			manager.channelSizes[n] = sz;
			manager.channelPtrs[n] = ptr;
			manager.channelShiftOffsets[n] = offset;
			manager.channelShiftTypes[n] = typesize;

			packedSize_byte += sz;
			n++;
		};

		for (const auto& kv : manager.getDataMap())
		{
			if (manager.needExchange(kv.first))
			{
				int sz = kv.second->datatype_size();

				if (sz % sizeof(int) != 0)
					die("Size of extra data per particle should be divisible by 4 bytes (PV '%s', data entry '%s')",
							ov->name.c_str(), kv.first.c_str());

				registerChannel(
						sz,
						reinterpret_cast<char*>(kv.second->genericDevPtr()),
						manager.shiftOffsetType(kv.first).first,
						manager.shiftOffsetType(kv.first).second  );
			}
		}

		if (upload)
		{
			manager.channelPtrs.        uploadToDevice(stream);
			manager.channelSizes.       uploadToDevice(stream);
			manager.channelShiftOffsets.uploadToDevice(stream);
			manager.channelShiftTypes.  uploadToDevice(stream);
		}

		channelData         = manager.channelPtrs.        devPtr();
		channelSizes        = manager.channelSizes.       devPtr();
		channelShiftOffsets = manager.channelShiftOffsets.devPtr();
		channelShiftTypes   = manager.channelShiftTypes.  devPtr();
	}
};


/**
 * Class that uses both ParticlePacker and ObjectExtraPacker
 * to pack everything. Provides totalPackedSize_byte of an object
 */
struct ObjectPacker
{
	ParticlePacker    part;
	ObjectExtraPacker obj;
	int totalPackedSize_byte = 0;

	ObjectPacker(ObjectVector* ov = nullptr, LocalObjectVector* lov = nullptr, cudaStream_t stream = 0) :
		part(ov, lov, stream), obj(ov, lov, stream)
	{
		if (ov == nullptr || lov == nullptr) return;
		totalPackedSize_byte = part.packedSize_byte * lov->nObjects + obj.packedSize_byte;
	}
};










