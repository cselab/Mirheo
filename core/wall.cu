#include <fstream>
#include <cmath>
#include <texture_types.h>

#include <core/helper_math.h>
#include <core/wall.h>
#include <core/celllist.h>
#include <core/interactions.h>
#include <core/interaction_engine.h>


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
   const float a0 = y[3] - y[2] - y[0] + y[1];
   const float a1 = y[0] - y[1] - a0;
   const float a2 = y[2] - y[0];
   const float a3 = y[1];

   return ((a0*mu + a1)*mu + a2)*mu + a3;
}


__global__ void cubicInterpolate3D(const float* in, int3 inDims, float3 inH, float* out, int3 outDims, float3 outH, float3 offset, float scaling)
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

				vals[dx+1] = in[ (curCoos.z*inDims.y + curCoos.y) * inDims.x + curCoos.x ] * scaling;
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
__device__ __forceinline__ float evalSdf(cudaTextureObject_t sdfTex, T x, float3 subDomainSize, float3 h, float3 invH)
{
	float3 x3{x.x, x.y, x.z};
	float3 texcoord = floorf((x3 + subDomainSize*0.5f) * invH);
	float3 lambda = (x3 - (texcoord * h - subDomainSize*0.5f)) * invH;

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


__global__ void countFrozen(const float4* pv, const int np, cudaTextureObject_t sdfTex, float3 subDomainSize, float3 h, int* nFrozen)
{
	const int pid = blockIdx.x * blockDim.x + threadIdx.x;
	if (pid >= np) return;

	const float4 coos = pv[2*pid];

	const float sdf = evalSdf(sdfTex, coos, subDomainSize, h, 1.0f / h);

	if (sdf > 0.0f && sdf < 1.2f)
	{
		atomicAggInc(nFrozen);
	}
}

__global__ void collectFrozen(cudaTextureObject_t sdfTex, float3 subDomainSize, float3 h, const int np,
		const float4* input, float4* remaining, float4* frozen, int* nRemaining, int* nFrozen)
{
	const int pid = blockIdx.x * blockDim.x + threadIdx.x;
	if (pid >= np) return;

	const float4 coos = input[2*pid];
	const float4 vels = input[2*pid+1];

	const float sdf = evalSdf(sdfTex, coos, subDomainSize, h, 1.0f / h);

	if (sdf <= 0.0f)
	{
		const int ind = atomicAggInc(nRemaining);
		remaining[2*ind] = coos;
		remaining[2*ind + 1] = vels;
	}

	if (sdf > 0.0f && sdf < 1.2f)
	{
		const int ind = atomicAggInc(nFrozen);
		frozen[2*ind] = coos;
		frozen[2*ind + 1] = vels;
	}
}

__global__ void countBoundaryCells(CellListInfo cinfo, cudaTextureObject_t sdfTex,
		const float3 subDomainSize, const float3 h, int* nBoundaryCells)
{
	const int cid = blockIdx.x * blockDim.x + threadIdx.x;
	if (cid >= cinfo.totcells) return;
	int ix, iy, iz;

	cinfo.decode(cid, ix, iy, iz);

	const float3 invH = 1.0f / h;

	const float cx = cinfo.domainStart.x + ix*cinfo.rc;
	const float cy = cinfo.domainStart.y + iy*cinfo.rc;
	const float cz = cinfo.domainStart.z + iz*cinfo.rc;

	const float l = cinfo.rc;
	const float s000 = evalSdf(sdfTex, make_float3(cx,   cy,   cz),   subDomainSize, h, invH);
	const float s001 = evalSdf(sdfTex, make_float3(cx,   cy,   cz+l), subDomainSize, h, invH);
	const float s010 = evalSdf(sdfTex, make_float3(cx,   cy+l, cz),   subDomainSize, h, invH);
	const float s011 = evalSdf(sdfTex, make_float3(cx,   cy+l, cz+l), subDomainSize, h, invH);
	const float s100 = evalSdf(sdfTex, make_float3(cx+l, cy,   cz),   subDomainSize, h, invH);
	const float s101 = evalSdf(sdfTex, make_float3(cx+l, cy,   cz+l), subDomainSize, h, invH);
	const float s110 = evalSdf(sdfTex, make_float3(cx+l, cy+l, cz),   subDomainSize, h, invH);
	const float s111 = evalSdf(sdfTex, make_float3(cx+l, cy+l, cz+l), subDomainSize, h, invH);

	if ( (0.1f > s000 && s000 > -1.1f) || (0.1f > s001 && s001 > -1.1f) ||
		 (0.1f > s010 && s010 > -1.1f) || (0.1f > s011 && s011 > -1.1f) ||
		 (0.1f > s100 && s100 > -1.1f) || (0.1f > s101 && s101 > -1.1f) ||
		 (0.1f > s110 && s110 > -1.1f) || (0.1f > s111 && s111 > -1.1f) )
	{
		atomicAggInc(nBoundaryCells);
	}
}

__global__ void getBoundaryCells(CellListInfo cinfo, cudaTextureObject_t sdfTex,
		const float3 subDomainSize, const float3 h, int* nBoundaryCells, int* boundaryCells)
{
	const int cid = blockIdx.x * blockDim.x + threadIdx.x;
	if (cid >= cinfo.totcells) return;

	int ix, iy, iz;

	cinfo.decode(cid, ix, iy, iz);

	const float3 invH = 1.0f / h;

	const float cx = cinfo.domainStart.x + ix*cinfo.rc;
	const float cy = cinfo.domainStart.y + iy*cinfo.rc;
	const float cz = cinfo.domainStart.z + iz*cinfo.rc;

	const float l = cinfo.rc;
	const float s000 = evalSdf(sdfTex, make_float3(cx,   cy,   cz),   subDomainSize, h, invH);
	const float s001 = evalSdf(sdfTex, make_float3(cx,   cy,   cz+l), subDomainSize, h, invH);
	const float s010 = evalSdf(sdfTex, make_float3(cx,   cy+l, cz),   subDomainSize, h, invH);
	const float s011 = evalSdf(sdfTex, make_float3(cx,   cy+l, cz+l), subDomainSize, h, invH);
	const float s100 = evalSdf(sdfTex, make_float3(cx+l, cy,   cz),   subDomainSize, h, invH);
	const float s101 = evalSdf(sdfTex, make_float3(cx+l, cy,   cz+l), subDomainSize, h, invH);
	const float s110 = evalSdf(sdfTex, make_float3(cx+l, cy+l, cz),   subDomainSize, h, invH);
	const float s111 = evalSdf(sdfTex, make_float3(cx+l, cy+l, cz+l), subDomainSize, h, invH);

	if ( (0.1f > s000 && s000 > -1.1f) || (0.1f > s001 && s001 > -1.1f) ||
		 (0.1f > s010 && s010 > -1.1f) || (0.1f > s011 && s011 > -1.1f) ||
		 (0.1f > s100 && s100 > -1.1f) || (0.1f > s101 && s101 > -1.1f) ||
		 (0.1f > s110 && s110 > -1.1f) || (0.1f > s111 && s111 > -1.1f) )
	{
		int id = atomicAggInc(nBoundaryCells);
		boundaryCells[id] = cid;
	}
}

__launch_bounds__(128, 8)
__global__ void bounceKernel(const int* wallCells, const int nWallCells, const int* __restrict__ cellsStart, CellListInfo cinfo, const float4* accs,
		cudaTextureObject_t sdfTex, const float3 subDomainSize, const float3 h, const float3 invH, float4* xyzouvwo, const float dt)
{
	const int maxNIters = 20;
	const float tolerance = 5e-6;

	const int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= nWallCells) return;
	const int cid = wallCells[tid];

	const int2 startSize = cinfo.decodeStartSize(cellsStart[cid]);

	for (int pid = startSize.x; pid < startSize.x + startSize.y; pid++)
	{
		float va, vb;

		float4 coo = xyzouvwo[2*pid];
		float4 vel = xyzouvwo[2*pid+1];

		// Warning - this is only valid for VV
		float4 oldCoo = coo - dt*vel;

		vb = evalSdf(sdfTex, coo, subDomainSize, h, invH);
		if (vb < 0.0f) continue; // if inside - continue

		va = evalSdf(sdfTex, oldCoo, subDomainSize, h, invH);
		assert( va < 0.0f ); // Accuracy issues here!

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
			vmid = evalSdf(sdfTex, mid, subDomainSize, h, invH);

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
		assert(fabs(vmid) < tolerance);

		// Final intersection at old*alpha + new*(1-alpha)
		const float alpha = (oldCoo.x - mid.x) / (oldCoo.x - coo.x);

		// Travel along alpha*(new - old), then bounces back along -(1-alpha)*(new - old)
		float beta = 2*alpha - 1;

		// In the corners long bounce may place the particle into another wall
		// Need to find a safe step in that case
		float4 candidate = oldCoo + beta * (coo - oldCoo);

		for (int i=0; i<maxNIters; i++)
		{
			if ( (evalSdf(sdfTex, candidate, subDomainSize, h, invH)) < 0.0f ) break;

			beta *= 0.5;
			candidate = oldCoo - beta * (coo - oldCoo);
		}

		// Not sure why, but this assertion always fails
		// even though everything seems allright
		//assert(vcandidate < 1.0f);

		xyzouvwo[2*pid] = candidate;
		xyzouvwo[2*pid + 1] = -vel;
	}
}

__global__ void checkKernel(float4* data, const int n, cudaTextureObject_t sdfTex, const float3 subDomainSize, const float3 h, const float3 invH)
{
	const int pid = blockIdx.x * blockDim.x + threadIdx.x;
	if (pid >= n) return;

	float4 coo = data[2*pid];
	float v = evalSdf(sdfTex, coo, subDomainSize, h, invH);

	if (v > 0.0f)
		printf("CHECK! %d:  [%f %f %f] -> %f\n", __float_as_int(coo.w), coo.x, coo.y, coo.z, v);
}

void Wall::_check()
{
	for (auto& pv : particleVectors)
	{
		checkKernel<<< (pv->np + 127) / 128, 128 >>>( (float4*)pv->coosvels.devPtr(), pv->np, sdfTex, subDomainSize, sdfH, 1.0 / sdfH);
	}
}


/*
 * We only set a few params here
 */
Wall::Wall(std::string name, std::string sdfFileName, float3 sdfH,  float _creationTime) :
		name(name), sdfFileName(sdfFileName), sdfH(sdfH), _creationTime(_creationTime), frozen(name)
{ }

void Wall::attach(ParticleVector* pv, CellList* cl)
{
	particleVectors.push_back(pv);
	cellLists.push_back(cl);

	const int oldSize = nBoundaryCells.size();
	boundaryCells.resize(oldSize+1);

	nBoundaryCells.resize(oldSize+1);
	nBoundaryCells.hostPtr()[oldSize] = 0;
	nBoundaryCells.uploadToDevice();
	countBoundaryCells<<< (cl->totcells + 127) / 128, 128 >>> (cl->cellInfo(), sdfTex, subDomainSize, sdfH, nBoundaryCells.devPtr()+oldSize);
	nBoundaryCells.downloadFromDevice();

	info("Found %d boundary cells", nBoundaryCells.hostPtr()[oldSize]);
	boundaryCells[oldSize].resize(nBoundaryCells.hostPtr()[oldSize]);

	nBoundaryCells.hostPtr()[oldSize] = 0;
	nBoundaryCells.uploadToDevice();
	getBoundaryCells<<< (cl->totcells + 127) / 128, 128 >>> (cl->cellInfo(), sdfTex, subDomainSize, sdfH,
			nBoundaryCells.devPtr()+oldSize, boundaryCells[oldSize].devPtr());
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


void Wall::create(MPI_Comm& comm, float3 subDomainStart, float3 subDomainSize, float3 globalDomainSize, ParticleVector* pv, CellList* cl)
{
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

	std::vector<float> fullSdfData;
	// Read heavy data
	readSdf(fullSdfSize_byte, endHeader_byte, nranks, rank, fullSdfData);


	subDomainResolution = make_int3(ceilf(subDomainSize / sdfH));
	sdfH = subDomainSize / make_float3(subDomainResolution);

	// Find your relevant chunk of data
	const float3 scale3 = globalDomainSize / initialSdfExtent;
	if ( fabs(scale3.x - scale3.y) > 1e-5 || fabs(scale3.x - scale3.z) > 1e-5 )
		die("Sdf size and domain size mismatch");
	const float scale = (scale3.x + scale3.y + scale3.z) / 3;

	const int margin = 3; // +2 from cubic interpolation, +1 from possible round-off errors
	float3 initialH = globalDomainSize / make_float3(initialSdfResolution-1);

	const int3 startId = make_int3( floorf(subDomainStart / initialH) );
	const int3 endId   = make_int3( ceilf((subDomainStart + subDomainSize) / initialH) );

	float3 startInLocalCoord = make_float3(startId - margin)*initialH - (subDomainStart + 0.5*subDomainSize);
	//float3 endInLocalCoord   = make_float3(endId   + margin)*initialH - (subDomainStart + 0.5*subDomainSize);
	const int3 inputResolution = (endId - startId) + make_int3(2*margin);

	PinnedBuffer<float> inputSdfData ( inputResolution.x * inputResolution.y * inputResolution.z );
	auto inpSdfDataPtr = inputSdfData.hostPtr();

	for (int k = 0; k < inputResolution.z; k++)
		for (int j = 0; j < inputResolution.y; j++)
			for (int i = 0; i < inputResolution.x; i++)
			{
				const int origIx = (i+startId.x + initialSdfResolution.x) % initialSdfResolution.x;
				const int origIy = (j+startId.y + initialSdfResolution.y) % initialSdfResolution.y;
				const int origIz = (k+startId.z + initialSdfResolution.z) % initialSdfResolution.z;

				inpSdfDataPtr[ (k*inputResolution.y + j)*inputResolution.x + i ] =
						fullSdfData[ (origIz*initialSdfResolution.y + origIy)*initialSdfResolution.x + origIx ];
			}

	// Compute offset
	float3 offset = startInLocalCoord - 0.5*subDomainSize;

	// Interpolate
	sdfRawData.resize(subDomainResolution.x * subDomainResolution.y * subDomainResolution.z);

	dim3 threads(8, 8, 8);
	dim3 blocks((subDomainResolution.x+threads.x-1) / threads.x,
				(subDomainResolution.y+threads.y-1) / threads.y,
				(subDomainResolution.z+threads.z-1) / threads.z);

	inputSdfData.uploadToDevice();
	float lenScalingFactor = scale;
	cubicInterpolate3D<<< blocks, threads >>>(inputSdfData.devPtr(), inputResolution, initialH, sdfRawData.devPtr(), subDomainResolution, sdfH, offset, lenScalingFactor);

	// Redistance
	// Need 2 arrays for redistancing

//	DeviceBuffer<float> tmp(sdfData.size);
//	const float redistDt = 0.1;
//	for (float t = 0; t < 200; t+=redistDt)
//	{
//		redistance<<< blocks, threads >>>(sdfData.devPtr(), resolution, h, redistDt, tmp.devPtr());
//		containerSwap(sdfData, tmp);
//	}

	// Prepare array to be transformed into texture
	auto chDesc = cudaCreateChannelDesc<float>();
	CUDA_Check( cudaMalloc3DArray(&sdfArray, &chDesc, make_cudaExtent(subDomainResolution.x, subDomainResolution.y, subDomainResolution.z)) );

	cudaMemcpy3DParms copyParams = {};
	copyParams.srcPtr = make_cudaPitchedPtr((void*)sdfRawData.devPtr(), subDomainResolution.x*sizeof(float), subDomainResolution.y, subDomainResolution.z);
	copyParams.dstArray = sdfArray;
	copyParams.extent = make_cudaExtent(subDomainResolution.x, subDomainResolution.y, subDomainResolution.z);
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

	PinnedBuffer<int> nFrozen(1), nRemaining(1), nBoundaryCells(1);

	nFrozen.clear();
	countFrozen<<< (pv->np + 127) / 128, 128 >>>((float4*)pv->coosvels.devPtr(), pv->np, sdfTex, subDomainSize, sdfH, nFrozen.devPtr());
	nFrozen.downloadFromDevice();

	frozen.resize(nFrozen.hostPtr()[0]);
	info("Freezing %d pv", nFrozen.hostPtr()[0]);

	nFrozen.   clear();
	nRemaining.clear();
	collectFrozen<<< (pv->np + 127) / 128, 128 >>>(sdfTex, subDomainSize, sdfH, pv->np,
			(float4*)pv->coosvels.devPtr(), (float4*)pv->pingPongBuf.devPtr(), (float4*)frozen.coosvels.devPtr(),
			nRemaining.devPtr(), nFrozen.devPtr());
	nRemaining.downloadFromDevice();
	nFrozen.   downloadFromDevice();


	CUDA_Check( cudaStreamSynchronize(0) );
	containerSwap(pv->coosvels, pv->pingPongBuf);
	pv->resize(nRemaining.hostPtr()[0]);
	info("Keeping %d pv", nRemaining.hostPtr()[0]);

	CUDA_Check( cudaDeviceSynchronize() );
}

void Wall::bounce(cudaStream_t stream)
{
	for (int i=0; i<particleVectors.size(); i++)
	{
		auto pv = particleVectors[i];
		auto cl = cellLists[i];

		bounceKernel<<< (boundaryCells[i].size() + 63) / 64, 64, 0, stream >>>(
				boundaryCells[i].devPtr(), boundaryCells[i].size(), cl->cellsStart.devPtr(), cl->cellInfo(), (float4*)pv->forces.devPtr(),
				sdfTex, subDomainSize, sdfH, 1.0 / sdfH, (float4*)pv->coosvels.devPtr(), dt);
	}
}

