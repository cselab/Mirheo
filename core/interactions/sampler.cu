#include "sampler.h"

#include <core/cuda_common.h>
#include <core/celllist.h>
#include <core/wall.h>
#include <core/cuda-rng.h>
#include "pairwise_engine.h"

//=============================================================================================
// SDF sampling from the wall
//=============================================================================================

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

//=============================================================================================
// Pairwise energy
//=============================================================================================

inline __device__ float fastPower(const float x, const float k)
{
	if (fabs(k - 1.0f)   < 1e-6f) return x;
	if (fabs(k - 0.5f)   < 1e-6f) return sqrtf(fabs(x));
	if (fabs(k - 0.25f)  < 1e-6f) return sqrtf(fabs(sqrtf(fabs(x))));
	if (fabs(k - 0.125f) < 1e-6f) return sqrtf(fabs(sqrtf(fabs(sqrtf(fabs(x))))));
	if (fabs(k - 2.0f)   < 1e-6f) return x*x;

    return powf(fabs(x), k);
}

__device__ __forceinline__ float E_DPD(
		Particle dst, Particle src,
		const float adpd, const float rc, const float rc2, const float invrc, const float k)
{
	const float3 dr = dst.r - src.r;
	const float rij2 = dot(dr, dr);
	if (rij2 > rc2) return 0.0f;

	const float invrij = rsqrtf(max(rij2, 1e-20f));
	const float rij = rij2 * invrij;
	const float argwr = 1.0f - rij*invrc;
	const float wr = fastPower(argwr, k+1);

	return adpd*rc / (k+1) * wr;
}

//=============================================================================================
// MCMC sampling kernel
//=============================================================================================

template<typename Potential>
__device__ __forceinline__ float E_inCell(Particle p, int3 cell,
		Particle* particles, CellListInfo cinfo, const uint* __restrict__ cellsStartSize,
		const float rc2, Potential potential)
{
	float E = 0.0f;
	for (int cellZ = max(cell.z-1, 0); cellZ <= min(cell.z+1, cinfo.ncells.z-1); cellZ++)
		for (int cellY = max(cell.y-1, 0); cellY <= min(cell.y+1, cinfo.ncells.y-1); cellY++)
			for (int cellX = max(cell.x-1, 0); cellX <= min(cell.x+1, cinfo.ncells.x-1); cellX++)
			{
				const int2 start_size = cinfo.decodeStartSize(cellsStartSize[cinfo.encode(cellX, cellY, cellZ)]);

				for (int othId = start_size.x; othId < start_size.x + start_size.y; othId++)
				{
					Particle othP = particles[othId];

					bool interacting = distance2(p.r, othP.r) < rc2;

					if (interacting && p.i1 != othP.i1)
						E += potential(p, othP);
				}
			}

	return E;
}

template<typename Potential>
__global__ void mcmcSample(int3 shift,
		Particle* particles, const int np, CellListInfo cinfo, const uint* __restrict__ cellsStartSize,
		const float rc2, const float kbT, const float seed,
		Wall::SdfInfo sdfInfo, const float minSdf, const float maxSdf,
		const float proposalFactor, int* nAccepted, int* nRejected, Potential potential)
{
	const uint3 uint3_blockDim{blockDim.x, blockDim.y, blockDim.z};
	const int3 cell0 = make_int3(blockIdx * uint3_blockDim + threadIdx) * make_int3(3) + shift;

	if (cell0.x >= cinfo.ncells.x || cell0.y >= cinfo.ncells.y || cell0.z >= cinfo.ncells.z)
		return;

	const int cid = cinfo.encode(cell0);
	const int2 start_size = cinfo.decodeStartSize(cellsStartSize[cid]);

	// Random particle, 3x random translations, acc probability = 5 rands
	const float rnd0 = Saru::uniform01(seed, cid, start_size.x);
	const float rnd1 = Saru::uniform01(rnd0, cid, start_size.x);
	const float rnd2 = Saru::uniform01(rnd1, cid, start_size.x);
	const float rnd3 = Saru::uniform01(rnd2, cid, start_size.x);
	const float rnd4 = Saru::uniform01(rnd3, cid, start_size.x);

	// Choose one random particle
	int pid = start_size.x + floorf(rnd0 * start_size.y);
	Particle p0 = particles[pid];

	// Just not the one initially in halo
	if (p0.i2 == 424242) return;

	// Propose a move
	Particle pnew = p0;
	pnew.r += proposalFactor * 2.0f*make_float3(rnd1 - 0.5f, rnd2 - 0.5f, rnd3 - 0.5f);
	int3 cell_new = cinfo.getCellIdAlongAxis(pnew.r);

	// Reject if the particle left sdf-bounded domain
	const float sdf = evalSdf(pnew.r, sdfInfo);
	if (sdf <= minSdf || sdf >= maxSdf)
	{
		atomicAggInc(nRejected);
		return;
	}

	// !!! COMPILER FUCKING ERROR SHISHISHI
	// LAMBDA KILLS NVCCCCCC !!1

	float E0   = E_inCell(p0,   cell0,    particles, cinfo, cellsStartSize, rc2, potential);
	float Enew = E_inCell(pnew, cell_new, particles, cinfo, cellsStartSize, rc2, potential);
	float dE = Enew - E0;

	// Accept if dE < 0 or with probability e^(-dE/kbT)
	if ( dE <= 0 || (dE > 0 && rnd4 < expf(-dE/kbT)) )
	{
		atomicAggInc(nAccepted);
		particles[pid] = pnew;
	}
	else
	{
		atomicAggInc(nRejected);
	}
}

//=============================================================================================
// Helper kernels
//=============================================================================================

__global__ void markHalo(Particle* particles, int np)
{
	const int gid = blockIdx.x * blockDim.x + threadIdx.x;
	if (gid >= np) return;

	particles[gid].i2 = 424242;
}

__global__ void writeBackLocal(Particle* srcs, int nSrc, Particle* dsts, int* nDst)
{
	const int gid = blockIdx.x * blockDim.x + threadIdx.x;
	if (gid >= nSrc) return;

	Particle p = srcs[gid];
	if (p.i2 != 424242)
	{
		int id = atomicAggInc(nDst);
		dsts[id] = p;
	}
}


MCMCSampler::MCMCSampler(pugi::xml_node node, Wall* wall) : nAccepted(1), nRejected(1), nDst(1), totE(1), wall(wall)
{
	name = node.attribute("name").as_string("");
	rc   = node.attribute("rc").as_float(1.0f);

	power = node.attribute("power").as_float(1.0f);
	a     = node.attribute("a")    .as_float(50);
	kbT   = node.attribute("kbt")  .as_float(1.0);

	combined = new ParticleVector("combined");
	combinedCL = nullptr;

	proposalFactor = 0.2f*rc;
}

void MCMCSampler::_compute(InteractionType type, ParticleVector* pv1, ParticleVector* pv2, CellList* cl, const float t, cudaStream_t stream)
{
	const int nthreads = 128;

	if (type != InteractionType::Halo || pv1 != pv2)
		return;

	auto pv = pv1;
	if (pv->local()->size() == 0)
		return;

	// Initialize combinedCL, if not yet
	// Make combinedCL include halo as well
	if (combinedCL == nullptr)
	{
		combined->domainSize = pv->domainSize;
		combined->globalDomainStart = pv->globalDomainStart;
		combinedCL = new PrimaryCellList(combined, rc, pv->domainSize + make_float3(2.0f));
	}


	// Copy halo from globalPV to the localPV
	int nLocal = pv->local()->size();
	int nHalo  = pv->halo()->size();
	combined->local()->resize(nLocal + nHalo, stream);
	combined->local()->changedStamp++;

	CUDA_Check( cudaMemcpyAsync(combined->local()->coosvels.devPtr(), pv->halo()->coosvels.devPtr(),
			nHalo * sizeof(Particle), cudaMemcpyDeviceToDevice, stream) );

	//   mark halo particleS
	markHalo<<< getNblocks(nHalo, nthreads), nthreads, 0, stream >>>(combined->local()->coosvels.devPtr(), pv->halo()->size());

	CUDA_Check( cudaMemcpyAsync(combined->local()->coosvels.devPtr() + nHalo, pv->local()->coosvels.devPtr(),
			pv->local()->size() * sizeof(Particle), cudaMemcpyDeviceToDevice, stream) );

	// Make cells
	combinedCL->build(stream);

	// Perform 27 MC sweeps
	float rc2 = rc*rc;
	float invrc = 1.0 / rc;
	float _rc = rc;
	float _a = a;
	float _power = power;
	auto potential = [=] __device__ (const Particle p1, const Particle p2) {
		return E_DPD(p1, p2, _a, _rc, rc2, invrc, _power);
	};

	dim3 threads3(4, 4, 4);
	dim3 blocks3;
	blocks3.x = getNblocks((combinedCL->ncells.x+2)/3, threads3.x);
	blocks3.y = getNblocks((combinedCL->ncells.y+2)/3, threads3.y);
	blocks3.z = getNblocks((combinedCL->ncells.z+2)/3, threads3.z);

	nAccepted.clear(stream);
	nRejected.clear(stream);
	for (int sx=0; sx<3; sx++)
		for (int sy=0; sy<3; sy++)
			for (int sz=0; sz<3; sz++)
			{
				int3 shift = make_int3(sx, sy, sz);

				mcmcSample <<< blocks3, threads3, 0, stream >>> (
						shift,
						combinedCL->coosvels->devPtr(), nLocal+nHalo, combinedCL->cellInfo(), combinedCL->cellsStartSize.devPtr(),
						rc2, kbT, drand48(),
						wall->sdfInfo, minSdf, maxSdf,
						proposalFactor, nAccepted.devPtr(), nRejected.devPtr(), potential);

				cudaDeviceSynchronize();
			}

	// Update the proposalFactor
	nAccepted.downloadFromDevice(stream);
	nRejected.downloadFromDevice(stream);
	float prob = (float)nAccepted[0] / (nAccepted[0] + nRejected[0]);

	debug("MCMC yielded %f acceptance probability, proposal scaling factor changed to %f", prob, proposalFactor);

	// Compute the total energy
	double old = totE[0];
	totE.clear(0);
	auto totalEptr = totE.devPtr();

	auto totEinter = [=] __device__ (const Particle p1, const Particle p2) {
		float E = E_DPD(p1, p2, _a, _rc, rc2, invrc, _power);
		atomicAdd(totalEptr, E);
		return make_float3(E, 0, 0);
	};
	combinedCL->forces->clear(stream);
	computeSelfInteractions<<< (nLocal+nHalo + nthreads - 1) / nthreads, nthreads, 0, stream >>>(
			nLocal+nHalo, (float4*)combinedCL->coosvels->devPtr(), (float*)combinedCL->forces->devPtr(),
			combinedCL->cellInfo(), combinedCL->cellsStartSize.devPtr(), rc*rc, totEinter);

	totE.downloadFromDevice(stream);
	printf("Total energy: %f, difference: %f\n", totE[0], totE[0] - old);

	// KILL ALL HUMANS
	/* piu piu piu */

	// Copy back the particles
	nDst.clear(stream);
	writeBackLocal<<< getNblocks(nLocal+nHalo, nthreads), nthreads, 0, stream >>>(
			combinedCL->coosvels->devPtr(), nLocal+nHalo, pv->local()->coosvels.devPtr(), nDst.devPtr());

	// Mark globalPV as changed
	pv->local()->changedStamp++;
}
