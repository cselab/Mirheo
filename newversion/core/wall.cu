#include "wall.h"
#include "flows.h"
#include "celllist.h"
#include "interactions.h"
#include "interaction_engine.h"

#include <fstream>
#include <cmath>
#include <helper_math.h>
#include <texture_types.h>

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

inline __host__ __device__ float3 ceilf(float3 v)
{
    return make_float3(ceilf(v.x), ceilf(v.y), ceilf(v.z));
}
//***************************************************************


__device__ __forceinline__ float cubicInterpolate1D(float y[4], float mu)
{
   const float a0 = y[3] - y[2] - y[0] + y[1];
   const float a1 = y[0] - y[1] - a0;
   const float a2 = y[2] - y[0];
   const float a3 = y[1];

   return ((a0*mu + a1)*mu + a2)*mu + a3;
}


__global__ void cubicInterpolate3D(const float* in, int3 inDims, float3 inH, float* out, int3 outDims, float3 outH, float3 offset)
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
	float3 tmp = make_float3(ix, iy, iz);
	float3 coos = tmp*outH + offset;


	// Reference point of the original grid
	int3 closestInPoint = min( make_int3( fmaxf( floorf(coos / inH), make_float3(0.0f)) ),  inDims-1);

	// Interpolate along x
	for (int dz = -1; dz <= 2; dz++)
		for (int dy = -1; dy <= 2; dy++)
		{
			float vals[4];

			for (int dx = -1; dx <= 2; dx++)
			{
				int3 delta{dx, dy, dz};
				const int3 curCoos = (closestInPoint+delta + inDims) % inDims;

				vals[dx+1] = in[ (curCoos.z*inDims.y + curCoos.y) * inDims.x + curCoos.x ];
			}

			interp2D[dz+1][dy+1] = cubicInterpolate1D(vals, (coos.x - closestInPoint.x*inH.x) / inH.x);
		}

	// Interpolate along y
	for (int dz = 0; dz <= 3; dz++)
		interp1D[dz] = cubicInterpolate1D(interp2D[dz], (coos.y - closestInPoint.y*inH.y) / inH.y);

	// Interpolate along z
	out[ (iz*outDims.y + iy) * outDims.x + ix ] = cubicInterpolate1D(interp1D, (coos.z - closestInPoint.z*inH.z) / inH.z);
}

__forceinline__ __device__ float gRouyTourin(float a, float b, float c, float d, float e, float f)
{
	// Rouy-Tourin scheme
	// http://epubs.siam.org/doi/pdf/10.1137/0729053

	return sqrt(
			max( sqr(max(a, 0.0f)), sqr(min(b, 0.0f)) ) +
			max( sqr(max(c, 0.0f)), sqr(min(d, 0.0f)) ) +
			max( sqr(max(e, 0.0f)), sqr(min(f, 0.0f)) )
			);
}

__global__ void redistance(const float* in, int3 dims, float3 h, float dt, float* out)
{
	const int ix = blockIdx.x * blockDim.x + threadIdx.x;
	const int iy = blockIdx.y * blockDim.y + threadIdx.y;
	const int iz = blockIdx.z * blockDim.z + threadIdx.z;

	auto sqr  = [](float x) { return x*x; };

	auto encode = [=](int i, int j, int k) {
		i = (i+dims.x) % dims.x;
		j = (j+dims.y) % dims.y;
		k = (k+dims.z) % dims.z;

		return (k*dims.y + j) * dims.x + i;
	};

	const int id0 = encode(ix, iy, iz);

	const float u        = in[id0];

	if (fabs(u) < max(h.x, max(h.y, h.z)))
	{
		out[id0] = in[id0];
		return;
	}

	const float ux_minus = in[encode(ix-1, iy,   iz  )];
	const float ux_plus  = in[encode(ix+1, iy,   iz  )];
	const float uy_minus = in[encode(ix,   iy-1, iz  )];
	const float uy_plus  = in[encode(ix,   iy+1, iz  )];
	const float uz_minus = in[encode(ix,   iy,   iz-1)];
	const float uz_plus  = in[encode(ix,   iy,   iz+1)];

	const float dx_minus = (u - ux_minus) / h.x;
	const float dx_plus  = (ux_plus  - u) / h.x;
	const float dy_minus = (u - uy_minus) / h.y;
	const float dy_plus  = (uy_plus  - u) / h.y;
	const float dz_minus = (u - uz_minus) / h.z;
	const float dz_plus  = (uz_plus  - u) / h.z;

	const float grad = gRouyTourin(dx_minus, dx_plus, dy_minus, dy_plus, dz_minus, dz_plus);

//	if (grad < 0.6f)
//		printf("%f,  x %f %f %f,  y %f %f %f,  z %f %f %f,  %d %d %d\n", grad,
//				ux_minus, u, ux_plus,  uy_minus, u, uy_plus,  uz_minus, u, uz_plus, ix, iy, iz);

	const float S = fabs(u) / sqrt( u*u + sqr(grad * h.x) );
	out[id0] = u + S * dt * (1-grad);
}

template<typename T>
__device__ __forceinline__ float evalSdf(cudaTextureObject_t sdfTex, T x, float3 length, float3 h, float3 invH)
{
	float3 x3{x.x, x.y, x.z};
	float3 texcoord = floorf((x3 + length*0.5f) * invH);
	float3 lambda = (x3 - (texcoord * h - length*0.5f)) * invH;

	const float s000 = tex3D<float>(sdfTex, texcoord.x + 0, texcoord.y + 0, texcoord.z + 0);
	const float s001 = tex3D<float>(sdfTex, texcoord.x + 1, texcoord.y + 0, texcoord.z + 0);
	const float s010 = tex3D<float>(sdfTex, texcoord.x + 0, texcoord.y + 1, texcoord.z + 0);
	const float s011 = tex3D<float>(sdfTex, texcoord.x + 1, texcoord.y + 1, texcoord.z + 0);
	const float s100 = tex3D<float>(sdfTex, texcoord.x + 0, texcoord.y + 0, texcoord.z + 1);
	const float s101 = tex3D<float>(sdfTex, texcoord.x + 1, texcoord.y + 0, texcoord.z + 1);
	const float s110 = tex3D<float>(sdfTex, texcoord.x + 0, texcoord.y + 1, texcoord.z + 1);
	const float s111 = tex3D<float>(sdfTex, texcoord.x + 1, texcoord.y + 1, texcoord.z + 1);

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


namespace FreezingActions
{
	const int Freeze = 1;
	const int Keep   = 2;
	const int Remove = 3;
}

__global__ void countFrozen(float4* particles, cudaTextureObject_t sdfTex, float3 length, float3 h, int* nFrozen)
{
	const int pid = blockIdx.x * blockDim.x + threadIdx.x;

	const float4 coos = particles[2*pid];
	float4 vels = particles[2*pid+1];

	const float sdf = evalSdf(sdfTex, coos, length, h, 1.0f / h);

	if (sdf < 0.0f)			vels.w = __int_as_float(FreezingActions::Keep);  // keep
	else if (sdf > 1.5f)	vels.w = __int_as_float(FreezingActions::Remove);  // remove
	else
	{
		vels.w = __int_as_float(FreezingActions::Freeze);  // freeze
		atomicAggInc(nFrozen);
	}

	particles[2*pid+1].w = vels.w;
}

__global__ void collectFrozen(cudaTextureObject_t sdfTex, const float4* input, float4* output, float4* frozen, int* nRemaining, int* nFrozen)
{
	const int pid = blockIdx.x * blockDim.x + threadIdx.x;

	const float4 coos = input[2*pid];
	const float4 vels = input[2*pid+1];

	const int key = __float_as_int(vels.w);

	if (key == FreezingActions::Keep)
	{
		const int ind = atomicAggInc(nRemaining);
		output[2*ind] = coos;
		output[2*ind + 1] = vels;
	}

	if (key == FreezingActions::Freeze)
	{
		const int ind = atomicAggInc(nFrozen);
		frozen[2*ind] = coos;
		frozen[2*ind + 1] = vels;
	}
}

__global__ void countBoundaryCells(const int3 ncells, const float3 domainStart, const float rc, cudaTextureObject_t sdfTex,
		const float3 length, const float3 h, int* nBoundaryCells)
{
	const int cid = blockIdx.x * blockDim.x + threadIdx.x;
	int ix, iy, iz;

	decode(cid, ix, iy, iz, ncells);

	const float3 invH = 1.0f / h;

	const float cx = domainStart.x + ix*rc - 1e-6f;
	const float cy = domainStart.y + iy*rc - 1e-6f;
	const float cz = domainStart.z + iz*rc - 1e-6f;

	const float l = rc+2e-6f;
	const float s000 = evalSdf(sdfTex, make_float3(cx,   cy,   cz),   length, h, invH);
	const float s001 = evalSdf(sdfTex, make_float3(cx,   cy,   cz+l), length, h, invH);
	const float s010 = evalSdf(sdfTex, make_float3(cx,   cy+l, cz),   length, h, invH);
	const float s011 = evalSdf(sdfTex, make_float3(cx,   cy+l, cz+l), length, h, invH);
	const float s100 = evalSdf(sdfTex, make_float3(cx+l, cy,   cz),   length, h, invH);
	const float s101 = evalSdf(sdfTex, make_float3(cx+l, cy,   cz+l), length, h, invH);
	const float s110 = evalSdf(sdfTex, make_float3(cx+l, cy+l, cz),   length, h, invH);
	const float s111 = evalSdf(sdfTex, make_float3(cx+l, cy+l, cz+l), length, h, invH);

	if ( (1e-6f > s000 && s000 > -1.000001f) || (1e-6f > s001 && s001 > -1.000001f) ||
		 (1e-6f > s010 && s010 > -1.000001f) || (1e-6f > s011 && s011 > -1.000001f) ||
		 (1e-6f > s100 && s100 > -1.000001f) || (1e-6f > s101 && s101 > -1.000001f) ||
		 (1e-6f > s110 && s110 > -1.000001f) || (1e-6f > s111 && s111 > -1.000001f) )
	{
		atomicAggInc(nBoundaryCells);
	}
}

__global__ void getBoundaryCells(const int3 ncells, const float3 domainStart, const float rc,
		cudaTextureObject_t sdfTex, const float3 length, const float3 h, int* nBoundaryCells, int* boundaryCells)
{
	const int cid = blockIdx.x * blockDim.x + threadIdx.x;
	int ix, iy, iz;

	decode(cid, ix, iy, iz, ncells);

	const float3 invH = 1.0f / h;

	const float cx = domainStart.x + ix*rc - 1e-6f;
	const float cy = domainStart.y + iy*rc - 1e-6f;
	const float cz = domainStart.z + iz*rc - 1e-6f;

	const float l = rc+2e-6f;
	const float s000 = evalSdf(sdfTex, make_float3(cx,   cy,   cz),   length, h, invH);
	const float s001 = evalSdf(sdfTex, make_float3(cx,   cy,   cz+l), length, h, invH);
	const float s010 = evalSdf(sdfTex, make_float3(cx,   cy+l, cz),   length, h, invH);
	const float s011 = evalSdf(sdfTex, make_float3(cx,   cy+l, cz+l), length, h, invH);
	const float s100 = evalSdf(sdfTex, make_float3(cx+l, cy,   cz),   length, h, invH);
	const float s101 = evalSdf(sdfTex, make_float3(cx+l, cy,   cz+l), length, h, invH);
	const float s110 = evalSdf(sdfTex, make_float3(cx+l, cy+l, cz),   length, h, invH);
	const float s111 = evalSdf(sdfTex, make_float3(cx+l, cy+l, cz+l), length, h, invH);

	if ( (1e-6f > s000 && s000 > -1.000001f) || (1e-6f > s001 && s001 > -1.000001f) ||
		 (1e-6f > s010 && s010 > -1.000001f) || (1e-6f > s011 && s011 > -1.000001f) ||
		 (1e-6f > s100 && s100 > -1.000001f) || (1e-6f > s101 && s101 > -1.000001f) ||
		 (1e-6f > s110 && s110 > -1.000001f) || (1e-6f > s111 && s111 > -1.000001f) )
	{
		int id = atomicAggInc(nBoundaryCells);
		boundaryCells[id] = cid;
	}
}


template<typename Transform>
__global__ void bounceBeforeIntegration(const int* wallCells, const int* __restrict__ cellsStart, const float4* accs,
		cudaTextureObject_t sdfTex, const float3 length, const float3 h, const float3 invH, float4* xyzouvwo, Transform transform)
{
	const int tid = blockIdx.x * blockDim.x + threadIdx.x;
	const int cid = wallCells[tid];

	const int2 startSize = decodeStartSize(cellsStart[cid]);

	for (int pid = startSize.x; pid < startSize.x + startSize.y; pid++)
	{
		float va, vb;

		float4 coo = xyzouvwo[2*pid];
		float4 vel = xyzouvwo[2*pid + 1];
		float4 acc = accs[pid];

		float4 oldCoo = coo;
		transform(coo, vel, acc, pid);

		va = evalSdf(sdfTex, coo, length, h, invH);
		if (va > 0.0f) continue;

		vb = evalSdf(sdfTex, oldCoo, length, h, invH);
		assert( vb >= 0.0f ); // Accuracy issues here!

		// Determine where we cross
		// Interpolation search

		float3 a{oldCoo.x, oldCoo.y, oldCoo.z};
		float3 b{coo.x, coo.y, coo.z};
		float vmid = 1.0f;
		float lambda;

		while (fabs(vmid) > 1e-6f)
		{
			lambda = vb / (vb - va);  // va*l + (1-l)*vb = 0
			const float3 mid = a*lambda + b*(1.0f - lambda);
			vmid = evalSdf(sdfTex, mid, length, h, invH);

			if (va * vmid < 0.0f)
				vb = vmid;
			else
				va = vmid;
		}

		// In the corners long bounce may place the particle into another wall
		// Need to find a safe step in that case
		float beta = 1-2*lambda;
		float4 candidate = oldCoo - beta * (coo - oldCoo);

		while (evalSdf(sdfTex, candidate, length, h, invH) > -1e-6f)
		{
			beta *= 0.5;
			candidate = oldCoo - beta * (coo - oldCoo);
		}

		xyzouvwo[2*pid] = candidate;
		xyzouvwo[2*pid + 1] = -vel;
	}
}


Wall::Wall(MPI_Comm& comm, IniParser& config): config(config)
{
	dt = config.getFloat("Common", "dt");

	std::string sdfname = config.getString("Wall", "SdfFileName");
	std::string velname = config.getString("Wall", "VelocityFileName", "");

	length = config.getFloat3("Common", "SubdomainSize");
	resolution = config.getInt3("Wall", "Resolution", make_int3(256));
	float floatMargin = config.getFloat("Wall", "margin");

	MPI_Check( MPI_Comm_dup(comm, &wallComm) );

	int nranks, rank;
	int ranks[3], periods[3], coords[3];
	MPI_Check( MPI_Comm_size(wallComm, &nranks) );
	MPI_Check( MPI_Comm_rank(wallComm, &rank) );
	MPI_Check( MPI_Cart_get (wallComm, 3, ranks, periods, coords) );

	int3 fullSdfResolution;
	float3 fullSdfExtent;
	int fullSdfSize;  // TODO int64_t
	int endHeader;

	// Read header
	if (rank == 0)
	{
		printf("'%s'\n", sdfname.c_str());
		std::ifstream file(sdfname);
		if (!file.good())
			die("File not found or not accessible");

		auto fstart = file.tellg();

		file >> fullSdfExtent.x >> fullSdfExtent.y >> fullSdfExtent.z >>
			fullSdfResolution.x >> fullSdfResolution.y >> fullSdfResolution.z;
		fullSdfSize = fullSdfResolution.x * fullSdfResolution.y * fullSdfResolution.z;

		info("Using wall file '%s' of size %.2fx%f.2x%f.2 and resolution %dx%dx%d", sdfname.c_str(),
				fullSdfExtent.x, fullSdfExtent.y, fullSdfExtent.z,
				fullSdfResolution.x, fullSdfResolution.y, fullSdfResolution.z);

		file.seekg( 0, std::ios::end );
		auto fend = file.tellg();

		endHeader = (int)(fend - fstart) - fullSdfSize * sizeof(float);

		file.close();
	}

	MPI_Check( MPI_Bcast(&fullSdfExtent,     3, MPI_FLOAT, 0, wallComm) );
	MPI_Check( MPI_Bcast(&fullSdfResolution, 3, MPI_INT,   0, wallComm) );
	MPI_Check( MPI_Bcast(&endHeader,         1, MPI_INT,   0, wallComm) );

	// Read part and allgather
	// TODO int64_t
	const int nPerProc = (fullSdfSize + nranks - 1) / nranks;
	std::vector<float> readBuffer(nPerProc);

	// Limits in bytes
	const int start = sizeof(float) * nPerProc * rank + endHeader;
	const int end   = std::min( start + sizeof(float) * nPerProc, sizeof(float) * fullSdfSize + endHeader);

	MPI_File fh;
	MPI_Status status;
	MPI_Check( MPI_File_open(wallComm, sdfname.c_str(), MPI_MODE_RDONLY, MPI_INFO_NULL, &fh) );
	MPI_Check( MPI_File_read_at_all(fh, start, &readBuffer[0], end - start, MPI_BYTE, &status) );

	std::vector<float> fullSdfData(nPerProc * nranks);  // May be bigger than fullSdfSize, to make gather easier
	MPI_Check( MPI_Allgather(&readBuffer[0], nPerProc, MPI_FLOAT, &fullSdfData[0], nPerProc, MPI_FLOAT, wallComm) );

	// Find your relevant chunk of data

	const float3 scale3 = length / fullSdfExtent;

	if ( fabs(scale3.x - scale3.y) > 1e-5 || fabs(scale3.x - scale3.z) > 1e-5 )
		die("Sdf size and domain size mismatch");

	const float scale = (scale3.x + scale3.y + scale3.z) / 3;

	float3 sdfH = scale * fullSdfExtent / make_float3(fullSdfResolution-1);
	float3 domainStart{length.x*coords[0], length.x*coords[1], length.x*coords[2]};  // TODO get it from settings

	int margin = 3; // +2 from cubic interpolation, +1 from possible round-off errors
	const int3 inputResolution = fullSdfResolution + make_int3(2*margin);

	int3 inputStart = make_int3( floorf(domainStart / sdfH) );

	PinnedBuffer<float> inputSdfData ( inputResolution.x * inputResolution.y * inputResolution.z );

	for (int k = 0; k < inputResolution.z; k++)
		for (int j = 0; j < inputResolution.y; j++)
			for (int i = 0; i < inputResolution.x; i++)
			{
				const int origIx = (i+inputStart.x-margin + fullSdfResolution.x) % fullSdfResolution.x;
				const int origIy = (j+inputStart.y-margin + fullSdfResolution.y) % fullSdfResolution.y;
				const int origIz = (k+inputStart.z-margin + fullSdfResolution.z) % fullSdfResolution.z;

				inputSdfData[ (k*inputResolution.y + j)*inputResolution.x + i ] =
						fullSdfData[ (origIz*fullSdfResolution.y + origIy)*fullSdfResolution.x + origIx ];
			}
	inputSdfData.synchronize(synchronizeDevice);

	// Compute offset
	float3 offset = margin*sdfH;

	// Interpolate
	sdfRawData.resize(resolution.x * resolution.y * resolution.z);

	h = length / make_float3(resolution-1);
	dim3 threads(8, 8, 8);
	dim3 blocks((resolution.x+threads.x-1) / threads.x, (resolution.y+threads.y-1) / threads.y, (resolution.z+threads.z-1) / threads.z);

	cubicInterpolate3D<<< blocks, threads >>>(inputSdfData.devdata, inputResolution, sdfH, sdfRawData.devdata, resolution, h, offset);

	// Redistance
	// Need 2 arrays for redistancing

//	DeviceBuffer<float> tmp(sdfData.size);
//	const float redistDt = 0.1;
//	for (float t = 0; t < 200; t+=redistDt)
//	{
//		redistance<<< blocks, threads >>>(sdfData.devdata, resolution, h, redistDt, tmp.devdata);
//		swap(sdfData, tmp);
//	}

	// Prepare array to be transformed into texture
	auto chDesc = cudaCreateChannelDesc<float>();
	CUDA_Check( cudaMalloc3DArray(&sdfArray, &chDesc, make_cudaExtent(resolution.x, resolution.y, resolution.z)) );

	cudaMemcpy3DParms copyParams = {};
	copyParams.srcPtr = make_cudaPitchedPtr(sdfRawData.devdata, resolution.x*sizeof(float), resolution.y, resolution.z);
	copyParams.dstArray = sdfArray;
	copyParams.extent = make_cudaExtent(resolution.x, resolution.y, resolution.z);
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

	CUDA_Check( cudaCreateTextureObject(&sdfTex, &resDesc, &texDesc, nullptr) );
}

void Wall::attach(ParticleVector* pv)
{
	particleVectors.push_back(pv);
}

void Wall::create(ParticleVector& dpds)
{
	PinnedBuffer<int> nFrozen(1), nRemaining(1), nBoundaryCells(1);

	nFrozen.clear();
	countFrozen<<< (dpds.np + 127) / 128, 128 >>>((float4*)dpds.coosvels.devdata, sdfTex, length, h, nFrozen.devdata);

	nFrozen.synchronize(synchronizeHost);
	frozen.resize(nFrozen[0]);

	nFrozen.   clear();
	nRemaining.clear();
	collectFrozen<<< (dpds.np + 127) / 128, 128 >>>(sdfTex, (float4*)dpds.coosvels.devdata, (float4*)dpds.pingPongBuf.devdata,
			(float4*)frozen.devdata, nRemaining.devdata, nFrozen.devdata);

	nRemaining.synchronize(synchronizeHost);
	dpds.resize(nRemaining[0]);
	swap(dpds.coosvels, dpds.pingPongBuf);


	nBoundaryCells.clear();
	countBoundaryCells<<< (dpds.totcells + 127) / 128, 128 >>> (dpds.ncells, dpds.domainStart, rc, sdfTex, length, h, nBoundaryCells.devdata);

	nBoundaryCells.synchronize(synchronizeHost);
	boundaryCells.resize(nBoundaryCells[0]);
	nBoundaryCells.clear();
	getBoundaryCells<<< (dpds.totcells + 127) / 128, 128 >>> (dpds.ncells, dpds.domainStart, rc, sdfTex, length, h, nBoundaryCells.devdata, boundaryCells.devdata);
}

void Wall::bounce(cudaStream_t stream)
{
	const float dt = this->dt;

	for (auto pv : particleVectors)
	{
		flowMacroWrapper( (bounceBeforeIntegration<<< (boundaryCells.size + 127) / 128, 128, 0, stream >>>(
				boundaryCells.devdata, pv->cellsStart.devdata, (float4*)pv->accs.devdata, sdfTex, length, h, 1.0 / h, (float4*)pv->coosvels.devdata, integrate)) );
	}
}

void Wall::computeInteractions(cudaStream_t stream)
{
	const float kBT = config.getFloat("Common", "kbt");
	const float gammadpd = config.getFloat("Common", "gamma");
	const float sigmadpd = sqrt(2 * gammadpd * kBT);
	const float adpd = config.getFloat("Common", "a");
	const float seed = 1.0f;

	const float sigma_dt = sigmadpd / sqrt(dt);
	auto dpdInt = [=] __device__ ( const float3 dstCoo, const float3 dstVel, const int dstId,
					   const float3 srcCoo, const float3 srcVel, const int srcId) {
		return dpd_interaction(dstCoo, dstVel, dstId, srcCoo, srcVel, srcId, adpd, gammadpd, sigma_dt, seed);
	};

	const int nth = 128;
	int i = 0;
	for (auto pv : particleVectors)
	{
		debug("Computing wall forces for %d-th particle vector", i++);
		computeExternalInteractions<false, true> <<< (frozen.size + nth - 1) / nth, nth, 0, stream >>>(
				(float4*)frozen.devdata, nullptr, (float4*)pv->coosvels.devdata, (float*)pv->accs.devdata, pv->cellsStart.devdata,
					pv->ncells, pv->domainStart, pv->totcells+1, frozen.size, dpdInt);
	}
}

