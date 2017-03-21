#include <fstream>
#include <cmath>
#include <texture_types.h>

#include <core/helper_math.h>
#include <core/wall.h>
#include <core/celllist.h>
#include <core/particle_vector.h>


// This should be in helper_math.h, but not there for some reason
//***************************************************************
inline __host__ __device__ int3 operator%(int3 a, int3 b)
{
    return make_int3(a.x % b.x, a.y % b.y, a.z % b.z);
}

inline __host__ __device__ int3 operator/(int3 a, int b)
{
    return make_int3(a.x / b, a.y / b, a.z / b);
}

//***************************************************************


__device__ __forceinline__ float cubicInterpolate1D(float y[4], float mu)
{
	// mu == 0 at y[1], mu == 1 at y[2]
	const float a0 = y[3] - y[2] - y[0] + y[1];
	const float a1 = y[0] - y[1] - a0;
	const float a2 = y[2] - y[0];
	const float a3 = y[1];

	return ((a0*mu + a1)*mu + a2)*mu + a3;
}


__global__ void cubicInterpolate3D(const float* in, int3 inDims, float3 inH, float* out, int3 outDims, float3 outH, float3 offset, float scalingFactor)
{
	// Inspired by http://paulbourke.net/miscellaneous/interpolation/
	// Center of the output domain is in offset
	// Center of the input domain is in (0,0,0)

	const int ix = blockIdx.x * blockDim.x + threadIdx.x;
	const int iy = blockIdx.y * blockDim.y + threadIdx.y;
	const int iz = blockIdx.z * blockDim.z + threadIdx.z;

	if (ix >= outDims.x || iy >= outDims.y || iz >= outDims.z) return;

	float interp2D[4][4];
	float interp1D[4];

	// Coordinates where to interpolate
	float3 outputId  = make_float3(ix, iy, iz);
	float3 outputCoo = outputId*outH;

	float3 inputCoo  = outputCoo + offset;

	// Make sure we're within the region where the the input data is defined
	assert( 0.0f <= inputCoo.x && inputCoo.x <= inDims.x*inH.x &&
			0.0f <= inputCoo.y && inputCoo.y <= inDims.y*inH.y &&
			0.0f <= inputCoo.z && inputCoo.z <= inDims.z*inH.z    );

	// Reference point of the original grid, rounded down
	int3 inputId_down = make_int3( floorf(inputCoo / inH) );
	float3 mu = (inputCoo - make_float3(inputId_down)*inH) / inH;

	// Interpolate along x
	for (int dz = -1; dz <= 2; dz++)
		for (int dy = -1; dy <= 2; dy++)
		{
			float vals[4];

			for (int dx = -1; dx <= 2; dx++)
			{
				int3 delta{dx, dy, dz};
				const int3 curInputId = (inputId_down+delta + inDims) % inDims;

				vals[dx+1] = in[ (curInputId.z*inDims.y + curInputId.y) * inDims.x + curInputId.x ] * scalingFactor;
			}

			interp2D[dz+1][dy+1] = cubicInterpolate1D(vals, mu.x);
		}

	// Interpolate along y
	for (int dz = 0; dz <= 3; dz++)
		interp1D[dz] = cubicInterpolate1D(interp2D[dz], mu.y);

	// Interpolate along z
	out[ (iz*outDims.y + iy) * outDims.x + ix ] = cubicInterpolate1D(interp1D, mu.z);
}

template<typename T>
__device__ __forceinline__ float evalSdf(T x, Wall::SdfInfo sdfInfo)
{
	float3 x3{x.x, x.y, x.z};
	float3 texcoord = floorf((x3 + sdfInfo.extendedDomainSize*0.5f) * sdfInfo.invh);
	float3 lambda = (x3 - (texcoord * sdfInfo.h - sdfInfo.extendedDomainSize*0.5f)) * sdfInfo.invh;

	const float s000 = tex3D<float>(sdfInfo.sdfTex, texcoord.x + 0, texcoord.y + 0, texcoord.z + 0);
	const float s001 = tex3D<float>(sdfInfo.sdfTex, texcoord.x + 1, texcoord.y + 0, texcoord.z + 0);
	const float s010 = tex3D<float>(sdfInfo.sdfTex, texcoord.x + 0, texcoord.y + 1, texcoord.z + 0);
	const float s011 = tex3D<float>(sdfInfo.sdfTex, texcoord.x + 1, texcoord.y + 1, texcoord.z + 0);
	const float s100 = tex3D<float>(sdfInfo.sdfTex, texcoord.x + 0, texcoord.y + 0, texcoord.z + 1);
	const float s101 = tex3D<float>(sdfInfo.sdfTex, texcoord.x + 1, texcoord.y + 0, texcoord.z + 1);
	const float s110 = tex3D<float>(sdfInfo.sdfTex, texcoord.x + 0, texcoord.y + 1, texcoord.z + 1);
	const float s111 = tex3D<float>(sdfInfo.sdfTex, texcoord.x + 1, texcoord.y + 1, texcoord.z + 1);

	const float s00x = s000 * (1 - lambda.x) + lambda.x * s001;
	const float s01x = s010 * (1 - lambda.x) + lambda.x * s011;
	const float s10x = s100 * (1 - lambda.x) + lambda.x * s101;
	const float s11x = s110 * (1 - lambda.x) + lambda.x * s111;

	const float s0yx = s00x * (1 - lambda.y) + lambda.y * s01x;
	const float s1yx = s10x * (1 - lambda.y) + lambda.y * s11x;

	const float szyx = s0yx * (1 - lambda.z) + lambda.z * s1yx;

//	printf("[%f %f %f]  [%f %f %f]  [%f %f %f]  = %f  vs  %f\n", x.x, x.y, x.z,  texcoord.x, texcoord.y, texcoord.z,
//			lambda.x, lambda.y, lambda.z, szyx, sqrt(x.x*x.x + x.y*x.y + x.z*x.z) - 5);

	return szyx;
}

// warp-aggregated atomic increment
// https://devblogs.nvidia.com/parallelforall/cuda-pro-tip-optimized-filtering-warp-aggregated-atomics/
// just4fun
__device__ __forceinline__ int atomicAggInc(int *ctr)
{
	int lane_id = (threadIdx.x % 32);

	int mask = __ballot(1);
	// select the leader
	int leader = __ffs(mask) - 1;
	// leader does the update
	int res;
	if(lane_id == leader)
	res = atomicAdd(ctr, __popc(mask));
	// broadcast result
	res = __shfl(res, leader);
	// each thread computes its own value
	return res + __popc(mask & ((1 << lane_id) - 1));
}


__global__ void countFrozen(const float4* pv, const int np, Wall::SdfInfo sdfInfo, int* nFrozen)
{
	const int pid = blockIdx.x * blockDim.x + threadIdx.x;
	if (pid >= np) return;

	const float4 coo = pv[2*pid];

	const float sdf = evalSdf(coo, sdfInfo);

	if (sdf > 0.0f && sdf < 1.2f)
	{
		atomicAggInc(nFrozen);
	}
}

__global__ void collectFrozen(const float4* input, const int np, Wall::SdfInfo sdfInfo,
		float4* remaining, float4* frozen, int* nRemaining, int* nFrozen)
{
	const int pid = blockIdx.x * blockDim.x + threadIdx.x;
	if (pid >= np) return;

	const float4 coo = input[2*pid];
	const float4 vel = input[2*pid+1];

	const float sdf = evalSdf(coo, sdfInfo);

	if (sdf <= 0.0f)
	{
		const int ind = atomicAggInc(nRemaining);
		remaining[2*ind] = coo;
		remaining[2*ind + 1] = vel;
	}

	if (sdf > 0.0f && sdf < 1.2f)
	{
		const int ind = atomicAggInc(nFrozen);
		frozen[2*ind] = coo;
		frozen[2*ind + 1] = make_float4(0.0f, 0.0f, 0.0f, vel.w);
	}
}

__device__ inline bool isCellOnBoundary(float3 cornerCoo, float3 len, Wall::SdfInfo sdfInfo)
{
	// About maximum distance a particle can cover in one step
	const float tol = 0.5f;

#pragma unroll
	for (int i=0; i<2; i++)
#pragma unroll
		for (int j=0; j<2; j++)
#pragma unroll
			for (int k=0; k<2; k++)
			{
				// Value in the cell corner
				const float3 shift = make_float3(i ? len.x : 0.0f, j ? len.y : 0.0f, k ? len.z : 0.0f);
				const float s = evalSdf( cornerCoo + shift,  sdfInfo );

				if (-1.0f - tol < s && s < 0.0f + tol)
					return true;
			}

	return false;
}

__global__ void countBoundaryCells(CellListInfo cinfo, Wall::SdfInfo sdfInfo, int* nBoundaryCells)
{
	const int cid = blockIdx.x * blockDim.x + threadIdx.x;
	if (cid >= cinfo.totcells) return;

	int3 ind;
	cinfo.decode(cid, ind.x, ind.y, ind.z);
	float3 cornerCoo = -0.5f*cinfo.domainSize + make_float3(ind)*cinfo.h;

	if (isCellOnBoundary(cornerCoo, cinfo.h, sdfInfo))
		atomicAggInc(nBoundaryCells);
}

__global__ void getBoundaryCells(CellListInfo cinfo, Wall::SdfInfo sdfInfo,
		int* nBoundaryCells, int* boundaryCells)
{
	const int cid = blockIdx.x * blockDim.x + threadIdx.x;
	if (cid >= cinfo.totcells) return;

	int3 ind;
	cinfo.decode(cid, ind.x, ind.y, ind.z);
	float3 cornerCoo = -0.5f*cinfo.domainSize + make_float3(ind)*cinfo.h;

	if (isCellOnBoundary(cornerCoo, cinfo.h, sdfInfo))
	{
		int id = atomicAggInc(nBoundaryCells);
		boundaryCells[id] = cid;
	}
}

__launch_bounds__(128, 8)
__global__ void bounceKernel(const int* wallCells, const int nWallCells, const uint* __restrict__ cellsStartSize, CellListInfo cinfo,
		Wall::SdfInfo sdfInfo, float4* coosvels, const float dt)
{
	const int maxNIters = 20;
	const float tolerance = 5e-6;

	const int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= nWallCells) return;
	const int cid = wallCells[tid];

	const int2 startSize = cinfo.decodeStartSize(cellsStartSize[cid]);

	for (int pid = startSize.x; pid < startSize.x + startSize.y; pid++)
	{
		float va, vb;

		float4 coo = coosvels[2*pid];
		float4 vel = coosvels[2*pid+1];

		// Warning - this is only valid for VV
		float4 oldCoo = coo - dt*vel;

		vb = evalSdf(coo, sdfInfo);

		if (vb < 0.0f) continue; // if inside - continue

		va = evalSdf(oldCoo, sdfInfo);

		// Accuracy issues here!
//		if (va > 0.0f)
//			printf("A particle %d: [%f %f %f] (%f)  -->  [%f %f %f] (%f) was inside already, bounce may fail\n",
//					__float_as_int(oldCoo.w), oldCoo.x, oldCoo.y, oldCoo.z, va, coo.x, coo.y, coo.z, vb);

		// Determine where we cross
		// Interpolation search

		float3 a{oldCoo.x, oldCoo.y, oldCoo.z};
		float3 b{coo.x, coo.y, coo.z};
		float3 mid;
		float vmid;

		int iters;
		for (iters=0; iters<maxNIters; iters++)
		{
			const float lambda = min(max((vb / (vb - va)), 0.01f), 0.99f);  // va*l + (1-l)*vb = 0
			mid = a*lambda + b*(1.0f - lambda);
			vmid = evalSdf(mid, sdfInfo);

			if (va * vmid < 0.0f)
			{
				vb = vmid;
				b = mid;
			}
			else
			{
				va = vmid;
				a = mid;
			}

			if (fabs(vmid) < tolerance) break;
		}

		// This shouldn't happen, but smth may go wrong
		if (fabs(vmid) > tolerance)
		{
			//printf("Solution was not found for bouncing!\n");
			coosvels[2*pid] = oldCoo;
			coosvels[2*pid + 1] = -vel;
			return;
		}

		// Final intersection at old*alpha + new*(1-alpha)
		// Take care if bounces are parallel to coo axes
		float alpha;
		if (oldCoo.x != coo.x)
			alpha = (oldCoo.x - mid.x) / (oldCoo.x - coo.x);
		else if (oldCoo.y != coo.y)
			alpha = (oldCoo.y - mid.y) / (oldCoo.y - coo.y);
		else if (oldCoo.z != coo.z)
			alpha = (oldCoo.z - mid.z) / (oldCoo.z - coo.z);
		else alpha = 1;

		// Travel along alpha*(new - old), then bounces back along -(1-alpha)*(new - old)
		float beta = 2*alpha - 1;

		// In the corners long bounce may place the particle into another wall
		// Need to find a safe step in that case
		float4 candidate = oldCoo + beta * (coo - oldCoo);

		for (int i=0; i<maxNIters; i++)
		{
			if ( (evalSdf(candidate, sdfInfo)) < 0.0f ) break;

			beta *= 0.5;
			candidate = oldCoo - beta * (coo - oldCoo);
		}

		if (evalSdf(candidate, sdfInfo) > 0.0f)
		{
//			printf("Bounce-back failed. Id %d: [%f %f %f] (%f)  -->  [%f %f %f] (%f)\n",
//					__float_as_int(oldCoo.w), oldCoo.x, oldCoo.y, oldCoo.z, va, coo.x, coo.y, coo.z, vb);

			coosvels[2*pid] = oldCoo;
			coosvels[2*pid + 1] = -vel;
			return;
		}

		coosvels[2*pid] = candidate;
		coosvels[2*pid + 1] = -vel;
	}
}

/*
 * We only set a few params here
 */
Wall::Wall(std::string name, std::string sdfFileName, float3 sdfH,  float _creationTime) :
		name(name), sdfFileName(sdfFileName), _creationTime(_creationTime)
{
	sdfInfo.h = sdfH;
	frozen = new ParticleVector(name);
}

void Wall::attach(ParticleVector* pv, CellList* cl)
{
	CUDA_Check( cudaDeviceSynchronize() );
	particleVectors.push_back(pv);
	cellLists.push_back(cl);

	const int oldSize = nBoundaryCells.size();
	boundaryCells.resize(oldSize+1);

	nBoundaryCells.resize(oldSize+1);
	nBoundaryCells.hostPtr()[oldSize] = 0;
	nBoundaryCells.uploadToDevice();
	countBoundaryCells<<< (cl->totcells + 127) / 128, 128 >>> (cl->cellInfo(), sdfInfo, nBoundaryCells.devPtr()+oldSize);
	nBoundaryCells.downloadFromDevice();

	debug("Found %d boundary cells", nBoundaryCells.hostPtr()[oldSize]);
	boundaryCells[oldSize].resize(nBoundaryCells.hostPtr()[oldSize]);

	nBoundaryCells.hostPtr()[oldSize] = 0;
	nBoundaryCells.uploadToDevice();
	getBoundaryCells<<< (cl->totcells + 127) / 128, 128 >>> (cl->cellInfo(), sdfInfo,
			nBoundaryCells.devPtr()+oldSize, boundaryCells[oldSize].devPtr());
	CUDA_Check( cudaDeviceSynchronize() );
}

void Wall::readHeader(int3& sdfResolution, float3& sdfExtent, int64_t& fullSdfSize_byte, int64_t& endHeader_byte, int rank)
{
	if (rank == 0)
	{
		//printf("'%s'\n", sdfFileName.c_str());
		std::ifstream file(sdfFileName);
		if (!file.good())
			die("File not found or not accessible");

		auto fstart = file.tellg();

		file >> sdfExtent.x >> sdfExtent.y >> sdfExtent.z >>
			sdfResolution.x >> sdfResolution.y >> sdfResolution.z;
		fullSdfSize_byte = (int64_t)sdfResolution.x * sdfResolution.y * sdfResolution.z * sizeof(float);

		info("Using wall file '%s' of size %.2fx%.2fx%.2f and resolution %dx%dx%d", sdfFileName.c_str(),
				sdfExtent.x, sdfExtent.y, sdfExtent.z,
				sdfResolution.x, sdfResolution.y, sdfResolution.z);

		file.seekg( 0, std::ios::end );
		auto fend = file.tellg();

		endHeader_byte = (fend - fstart) - fullSdfSize_byte;

		file.close();
	}

	MPI_Check( MPI_Bcast(&sdfExtent,        3, MPI_FLOAT,     0, wallComm) );
	MPI_Check( MPI_Bcast(&sdfResolution,    3, MPI_INT,       0, wallComm) );
	MPI_Check( MPI_Bcast(&fullSdfSize_byte, 1, MPI_INT64_T,   0, wallComm) );
	MPI_Check( MPI_Bcast(&endHeader_byte,   1, MPI_INT64_T,   0, wallComm) );
}

void Wall::readSdf(int64_t fullSdfSize_byte, int64_t endHeader_byte, int nranks, int rank, std::vector<float>& fullSdfData)
{
	// Read part and allgather
	const int64_t readPerProc_byte = (fullSdfSize_byte + nranks - 1) / (int64_t)nranks;
	std::vector<char> readBuffer(readPerProc_byte);

	// Limits in bytes
	const int64_t readStart = readPerProc_byte * rank + endHeader_byte;
	const int64_t readEnd   = std::min( readStart + readPerProc_byte, fullSdfSize_byte + endHeader_byte);

	MPI_File fh;
	MPI_Status status;
	MPI_Check( MPI_File_open(wallComm, sdfFileName.c_str(), MPI_MODE_RDONLY, MPI_INFO_NULL, &fh) );  // TODO: MPI_Info
	MPI_Check( MPI_File_read_at_all(fh, readStart, readBuffer.data(), readEnd - readStart, MPI_BYTE, &status) );
	// TODO: check that we read just what we asked
	// MPI_Get_count only return int though

	fullSdfData.resize(readPerProc_byte * nranks / sizeof(float));  // May be bigger than fullSdfSize, to make gather easier
	MPI_Check( MPI_Allgather(readBuffer.data(), readPerProc_byte, MPI_BYTE, fullSdfData.data(), readPerProc_byte, MPI_BYTE, wallComm) );
}

void Wall::prepareRelevantSdfPiece(const float* fullSdfData, float3 extendedDomainStart, float3 initialSdfH, int3 initialSdfResolution,
		int3& resolution, float3& offset, PinnedBuffer<float>& localSdfData)
{
	// Find your relevant chunk of data
	// We cannot send big sdf files directly, so we'll carve a piece now

	const int margin = 3; // +2 from cubic interpolation, +1 from possible round-off errors
	const int3 startId = make_int3( floorf( extendedDomainStart                             / initialSdfH) ) - margin;
	const int3 endId   = make_int3( ceilf ((extendedDomainStart+sdfInfo.extendedDomainSize) / initialSdfH) ) + margin;

	float3 startInLocalCoord = make_float3(startId)*initialSdfH - (extendedDomainStart + 0.5*sdfInfo.extendedDomainSize);
	offset = -0.5*sdfInfo.extendedDomainSize - startInLocalCoord;

	int rank;
	MPI_Check( MPI_Comm_rank(wallComm, &rank) );
//	printf("%d:  SDstart [%f %f %f]  sdfH [%f %f %f] startId [%d %d %d], endId [%d %d %d], localstart [%f %f %f]\n",
//				rank,
//				extendedDomainStart.x, extendedDomainStart.y, extendedDomainStart.z,
//				initialSdfH.x, initialSdfH.y, initialSdfH.z,
//				startId.x, startId.y, startId.z,
//				endId.x, endId.y, endId.z,
//				startInLocalCoord.x, startInLocalCoord.y, startInLocalCoord.z);

	resolution = endId - startId;

	localSdfData.resize( resolution.x * resolution.y * resolution.z );
	auto locSdfDataPtr = localSdfData.hostPtr();

//	printf("%d:  input [%d %d %d], initial [%d %d %d], start [%d %d %d]\n",
//			rank, resolution.x, resolution.y, resolution.z,
//			initialSdfResolution.x, initialSdfResolution.y, initialSdfResolution.z,
//			startId.x, startId.y, startId.z);

#warning "Minus here should be removed"
	for (int k = 0; k < resolution.z; k++)
		for (int j = 0; j < resolution.y; j++)
			for (int i = 0; i < resolution.x; i++)
			{
				const int origIx = (i+startId.x + initialSdfResolution.x) % initialSdfResolution.x;
				const int origIy = (j+startId.y + initialSdfResolution.y) % initialSdfResolution.y;
				const int origIz = (k+startId.z + initialSdfResolution.z) % initialSdfResolution.z;

				// FIXME: AAAAAAAAAAAAAAAAA MINUS
				locSdfDataPtr[ (k*resolution.y + j)*resolution.x + i ] = -
						fullSdfData[ (origIz*initialSdfResolution.y + origIy)*initialSdfResolution.x + origIx ];
			}
}

void Wall::createSdf(MPI_Comm& comm, float3 subDomainStart, float3 subDomainSize, float3 globalDomainSize)
{
	debug2("Creating wall");

	CUDA_Check( cudaDeviceSynchronize() );
	MPI_Check( MPI_Comm_dup(comm, &wallComm) );

	int nranks, rank;
	int ranks[3], periods[3], coords[3];
	MPI_Check( MPI_Comm_size(wallComm, &nranks) );
	MPI_Check( MPI_Comm_rank(wallComm, &rank) );
	MPI_Check( MPI_Cart_get (wallComm, 3, ranks, periods, coords) );

	int3 initialSdfResolution;
	float3 initialSdfExtent;

	int64_t fullSdfSize_byte;
	int64_t endHeader_byte;

	// Read header
	readHeader(initialSdfResolution, initialSdfExtent, fullSdfSize_byte, endHeader_byte, rank);
	float3 initialSdfH = globalDomainSize / make_float3(initialSdfResolution-1);

	// Read heavy data
	std::vector<float> fullSdfData;
	readSdf(fullSdfSize_byte, endHeader_byte, nranks, rank, fullSdfData);

	// We'll make sdf a bit bigger, so that particles that flew away
	// would also be correctly bounced back
	sdfInfo.extendedDomainSize = subDomainSize + 2.0f*margin3;
	sdfInfo.resolution         = make_int3( ceilf(sdfInfo.extendedDomainSize / sdfInfo.h) );
	sdfInfo.h                  = sdfInfo.extendedDomainSize / make_float3(sdfInfo.resolution-1);
	sdfInfo.invh               = 1.0f / sdfInfo.h;

	const float3 scale3 = globalDomainSize / initialSdfExtent;
	if ( fabs(scale3.x - scale3.y) > 1e-5 || fabs(scale3.x - scale3.z) > 1e-5 )
		die("Sdf size and domain size mismatch");
	const float lenScalingFactor = (scale3.x + scale3.y + scale3.z) / 3;

	int3 resolutionBeforeInterpolation;
	float3 offset;
	PinnedBuffer<float> localSdfData;
	prepareRelevantSdfPiece(fullSdfData.data(), subDomainStart - margin3, initialSdfH, initialSdfResolution,
			resolutionBeforeInterpolation, offset, localSdfData);

	// Interpolate
	sdfRawData.resize(sdfInfo.resolution.x * sdfInfo.resolution.y * sdfInfo.resolution.z);

	dim3 threads(8, 8, 8);
	dim3 blocks((sdfInfo.resolution.x+threads.x-1) / threads.x,
				(sdfInfo.resolution.y+threads.y-1) / threads.y,
				(sdfInfo.resolution.z+threads.z-1) / threads.z);

	localSdfData.uploadToDevice();
	cubicInterpolate3D<<< blocks, threads >>>(localSdfData.devPtr(), resolutionBeforeInterpolation, initialSdfH,
			sdfRawData.devPtr(), sdfInfo.resolution, sdfInfo.h, offset, lenScalingFactor);


	// Prepare array to be transformed into texture
	auto chDesc = cudaCreateChannelDesc<float>();
	CUDA_Check( cudaMalloc3DArray(&sdfArray, &chDesc, make_cudaExtent(sdfInfo.resolution.x, sdfInfo.resolution.y, sdfInfo.resolution.z)) );

	cudaMemcpy3DParms copyParams = {};
	copyParams.srcPtr = make_cudaPitchedPtr((void*)sdfRawData.devPtr(), sdfInfo.resolution.x*sizeof(float), sdfInfo.resolution.x, sdfInfo.resolution.y);
	copyParams.dstArray = sdfArray;
	copyParams.extent = make_cudaExtent(sdfInfo.resolution.x, sdfInfo.resolution.y, sdfInfo.resolution.z);
	copyParams.kind = cudaMemcpyDeviceToDevice;

	CUDA_Check( cudaMemcpy3D(&copyParams) );

	// Create texture
	cudaResourceDesc resDesc = {};
	resDesc.resType = cudaResourceTypeArray;
	resDesc.res.array.array = sdfArray;

	cudaTextureDesc texDesc = {};
	texDesc.addressMode[0]   = cudaAddressModeWrap;
	texDesc.addressMode[1]   = cudaAddressModeWrap;
	texDesc.addressMode[2]   = cudaAddressModeWrap;
	texDesc.filterMode       = cudaFilterModePoint;
	texDesc.readMode         = cudaReadModeElementType;
	texDesc.normalizedCoords = 0;

	CUDA_Check( cudaCreateTextureObject(&sdfInfo.sdfTex, &resDesc, &texDesc, nullptr) );
}

void Wall::freezeParticles(ParticleVector* pv)
{
	PinnedBuffer<int> nFrozen(1), nRemaining(1), nBoundaryCells(1);

	nFrozen.clear();
	countFrozen<<< (pv->np + 127) / 128, 128 >>>((float4*)pv->coosvels.devPtr(), pv->np, sdfInfo, nFrozen.devPtr());
	nFrozen.downloadFromDevice();

	frozen->resize(nFrozen.hostPtr()[0]);
	frozen->mass = pv->mass;
	frozen->domainLength = pv->domainLength;

	info("Freezing %d pv", nFrozen.hostPtr()[0]);

	nFrozen.   clear();
	nRemaining.clear();
	collectFrozen<<< (pv->np + 127) / 128, 128 >>>( (float4*)pv->coosvels.devPtr(), pv->np, sdfInfo,
			(float4*)pv->pingPongCoosvels.devPtr(), (float4*)frozen->coosvels.devPtr(),
			nRemaining.devPtr(), nFrozen.devPtr());
	nRemaining.downloadFromDevice();
	nFrozen.   downloadFromDevice();


	CUDA_Check( cudaStreamSynchronize(0) );
	containerSwap(pv->coosvels, pv->pingPongCoosvels);
	pv->resize(nRemaining.hostPtr()[0]);
	pv->changedStamp++;
	info("Keeping %d pv", nRemaining.hostPtr()[0]);

	CUDA_Check( cudaDeviceSynchronize() );
}

void Wall::bounce(float dt, cudaStream_t stream)
{
	for (int i=0; i<particleVectors.size(); i++)
	{
		auto pv = particleVectors[i];
		auto cl = cellLists[i];

		debug2("Bouncing %d %s particles", pv->size(), pv->name.c_str());
		bounceKernel<<< (boundaryCells[i].size() + 63) / 64, 64, 0, stream >>>(
				boundaryCells[i].devPtr(), boundaryCells[i].size(), cl->cellsStartSize.devPtr(), cl->cellInfo(),
				sdfInfo, (float4*)pv->coosvels.devPtr(), dt);
	}
}


__global__ void _check(float4* coosvels, int n, Wall::SdfInfo sdfInfo)
{
	const int pid = blockIdx.x * blockDim.x + threadIdx.x;
	if (pid >= n) return;

	float4 coo = coosvels[2*pid];
	float v = evalSdf(coo, sdfInfo);

	if (v > 0) printf("Check failed %d (in array %d): [%f %f %f] (%f)\n", __float_as_int(coo.w), pid, coo.x, coo.y, coo.z, v);
}

void Wall::check(Particle* parts, int n, cudaStream_t stream)
{
	_check<<< (n+127)/128, 128, 0, stream >>> ((float4*)parts, n, sdfInfo);
}

