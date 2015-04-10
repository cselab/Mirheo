/*
 *  cuda-dpd.cu
 *  Part of CTC/cuda-dpd-sem/dpd/
 *
 *  Evaluation of DPD force using Newton's 3rd law
 *  Created and authored by Yu-Hang Tang and Mauro Bisson on 2015-04-01.
 *  Copyright 2015. All rights reserved.
 *
 *  Users are NOT authorized
 *  to employ the present software for their own publications
 *  before getting a written permission from the author of this file.
 */

#include <cstdio>
#include <cassert>

#include "cuda-dpd.h"
#include "../dpd-rng.h"

struct InfoDPD
{
    int3 ncells;
    float3 domainsize, invdomainsize, domainstart;
    float invrc, aij, gamma, sigmaf;
    float * axayaz;
    float seed;
};

__constant__ InfoDPD info;

texture<float4, cudaTextureType1D> texParticlesF4;
texture<ushort4, cudaTextureType1D, cudaReadModeNormalizedFloat> texParticlesH4;
texture<uint2, cudaTextureType1D> texStartAndCount;
 
#define TRANSPOSED_ATOMICS
//#define ONESTEP
#define LETS_MAKE_IT_MESSY
#define CRAZY_SMEM

#define _XCPB_ 2
#define _YCPB_ 2
#define _ZCPB_ 1
#define CPB (_XCPB_ * _YCPB_ * _ZCPB_)

__device__ float3 _dpd_interaction(const int dpid, const float4 xdest, const float4 udest, const float4 xsrc, const float4 usrc, const int spid)
{
    const float _xr = xdest.x - xsrc.x;
    const float _yr = xdest.y - xsrc.y;
    const float _zr = xdest.z - xsrc.z;

    const float rij2 = _xr * _xr + _yr * _yr + _zr * _zr;
    assert(rij2 < 1);

    const float invrij = rsqrtf(rij2);
    const float rij = rij2 * invrij;
    const float wc = 1 - rij;
    const float wr = viscosity_function<-VISCOSITY_S_LEVEL>(wc);

    const float xr = _xr * invrij;
    const float yr = _yr * invrij;
    const float zr = _zr * invrij;

    const float rdotv =
	xr * (udest.x - usrc.x) +
	yr * (udest.y - usrc.y) +
	zr * (udest.z - usrc.z);

    const float myrandnr = Logistic::mean0var1(info.seed, min(spid, dpid), max(spid, dpid));

    const float strength = info.aij * wc - (info.gamma * wr * rdotv + info.sigmaf * myrandnr) * wr;

    return make_float3(strength * xr, strength * yr, strength * zr);
}

#define __IMOD(x,y) ((x)-((x)/(y))*(y))

__inline__ __device__ uint __lanemask_lt() {
	uint mask;
	asm("mov.u32 %0, %lanemask_lt;" : "=r"(mask) );
	return mask;
}

__inline__ __device__ uint __pack_8_24(uint a, uint b) {
	uint d;
	asm("bfi.b32  %0, %1, %2, 24, 8;" : "=r"(d) : "r"(a), "r"(b) );
	return d;
}

__inline__ __device__ uint2 __unpack_8_24(uint d) {
	uint a;
	asm("bfe.u32  %0, %1, 24, 8;" : "=r"(a) : "r"(d) ); // TODO: can I destructively extract?
	return make_uint2( a, d&0x00FFFFFFU );
}

__device__ char4 tid2ind[14] = {{-1, -1, -1, 0}, {0, -1, -1, 0}, {1, -1, -1, 0},
				{-1,  0, -1, 0}, {0,  0, -1, 0}, {1,  0, -1, 0},
				{-1,  1, -1, 0}, {0,  1, -1, 0}, {1,  1, -1, 0},
				{-1, -1,  0, 0}, {0, -1,  0, 0}, {1, -1,  0, 0},
				{-1,  0,  0, 0}, {0,  0,  0, 0}};
#define MYCPBX	(4)
#define MYCPBY	(2)
#define MYCPBZ	(2)
#define MYWPB	(4)

#ifdef CRAZY_SMEM
__forceinline__ __device__ void core_ytang(const uint dststart, const uint pshare, const uint tid, const uint spidext ) {
#else
__forceinline__ __device__ void core_ytang(volatile uint const *queue, const uint dststart, const uint wid, const uint tid, const uint spidext ) {
#endif

#ifdef CRAZY_SMEM
	uint item;
	const uint offset = xmad( tid, 4.f, pshare );
	asm("ld.volatile.shared.u32 %0, [%1+1024];" : "=r"(item) : "r"(offset) : "memory" );
	const uint2 pid = __unpack_8_24( item );
#else
	const uint2 pid = __unpack_8_24( queue[tid] );
#endif
	const uint dpid = xadd( dststart, pid.x );
	const uint spid = pid.y;

	const uint dentry = xscale( dpid, 2.f );
	const uint sentry = xscale( spid, 2.f );
	const float4 xdest = tex1Dfetch(texParticlesF4,       dentry       );
	const float4 xsrc  = tex1Dfetch(texParticlesF4,       sentry       );
	const float4 udest = tex1Dfetch(texParticlesF4, xadd( dentry, 1u ) );
	const float4 usrc  = tex1Dfetch(texParticlesF4, xadd( sentry, 1u ) );
	const float3 f = _dpd_interaction(dpid, xdest, udest, xsrc, usrc, spid);

	// the overhead of transposition acc back
	// can be completely killed by changing the integration kernel
	#ifdef TRANSPOSED_ATOMICS
	uint base = dpid & 0xFFFFFFE0U; // TODO: use xdiv + xscale?
	uint off  = xsub( dpid, base );
	float* acc = info.axayaz + xmad( base, 3.f, off );
	atomicAdd(acc   , f.x);
	atomicAdd(acc+32, f.y);
	atomicAdd(acc+64, f.z);

	if (spid < spidext) { // TODO: PTX bool
		uint base = spid & 0xFFFFFFE0U;
		uint off  = xsub( spid, base );
		float* acc = info.axayaz + xmad( base, 3.f, off );
		atomicAdd(acc   , -f.x);
		atomicAdd(acc+32, -f.y);
		atomicAdd(acc+64, -f.z);
	}
	#else
	float* acc = info.axayaz + xscale( dpid, 3.f );
	atomicAdd(acc  , f.x);
	atomicAdd(acc+1, f.y);
	atomicAdd(acc+2, f.z);

	if (spid < spidext) {
		float* acc = info.axayaz + xscale( spid, 3.f );
		atomicAdd(acc  , -f.x);
		atomicAdd(acc+1, -f.y);
		atomicAdd(acc+2, -f.z);
	}
	#endif
}

__global__  __launch_bounds__(32*MYWPB, 16)
void _dpd_forces_symm_merged() {

	asm volatile(".shared .u32 smem[512];" ::: "memory" );
	//* was: __shared__ uint2 volatile start_n_scan[MYWPB][32];
	//* was: __shared__ uint  volatile queue[MYWPB][64];

	const uint tid = threadIdx.x; // TODO: how to prevent S2R?
	const uint wid = threadIdx.y;
	const uint pshare = xscale( threadIdx.y, 256.f );

	const char4 offs = __ldg(tid2ind+tid);
	const int cbase = blockIdx.z*MYCPBZ*info.ncells.x*info.ncells.y +
                          blockIdx.y*MYCPBY*info.ncells.x +
                          blockIdx.x*MYCPBX + wid +
			  offs.z*info.ncells.x*info.ncells.y +
			  offs.y*info.ncells.x +
			  offs.x;

	//#pragma unroll 4 // faster on k20x, slower on k20
	for(int it = 3; it >= 0; it--) { // TODO: TINYFLOAT

		const int cid = cbase +
				(it>1)*info.ncells.x*info.ncells.y +
				((it&1)^((it>>1)&1))*info.ncells.x;

		uint mystart = 0, mycount = 0, myscan;
		asm("{  .reg .pred vc;"
			"   .reg .u32  foo, bar;"
			"   .reg .s64  texture;"
			"    setp.lt.f32     vc, %2, %3;"
			"    setp.ge.and.f32 vc, %5, 0.0, vc;"
			"    setp.lt.and.s32 vc, %4, %6, vc;"
			"    selp.s32 %0, 1, 0, vc;"
			"@vc mov.u64 texture, texStartAndCount;"
			"@vc tex.1d.v4.s32.s32 {%0, %1, foo, bar}, [texture, %4];"
			"}" :
			"=r"(mystart), "=r"(mycount)  :
			"f"(u2f(tid)), "f"(u2f(14u)), "r"(cid), "f"(i2f(cid)),
			"r"(info.ncells.x*info.ncells.y*info.ncells.z) );

		myscan  = mycount;
		asm("st.volatile.shared.u32 [%0], %1;" ::
			"r"( xmad( tid, 8.f, pshare ) ),
			"r"( mystart ) :
			"memory" );
		//* was: uint mycount=0, myscan=0;
		//* was: if (tid < 14) { // TODO: do tid<14 predicate later
			//* was: const int cid = cbase +
			//* was: 		(it>1)*info.ncells.x*info.ncells.y +
			//* was: 		((it&1)^((it>>1)&1))*info.ncells.x;

			//* was: const bool valid_cid = (cid >= 0) && (cid < info.ncells.x*info.ncells.y*info.ncells.z);
			//* was: const uint2 sc = valid_cid ? tex1Dfetch( texStartAndCount, cid ) : make_uint2(0,0);

			//* was: start_n_scan[wid][tid].x = (valid_cid) ? tex1Dfetch(texStart, cid) : 0;
			//* was: myscan = mycount = (valid_cid) ? tex1Dfetch(texCount, cid) : 0;
		//* was: }
	   
		// was: #pragma unroll
		// was: for(int L = 1; L < 32; L <<= 1) {
		// was: 	myscan = xadd( myscan, (tid >= L)*__shfl_up(myscan, L) );
		// was: }

		asm("{ .reg .pred   p;"
			"  .reg .f32    myscan, theirscan;"
			"   mov.b32     myscan, %0;"
			"   shfl.up.b32 theirscan|p, myscan, 0x1, 0x0;"
			"@p add.f32     myscan, theirscan, myscan;"
			"   shfl.up.b32 theirscan|p, myscan, 0x2, 0x0;"
			"@p add.f32     myscan, theirscan, myscan;"
			"   shfl.up.b32 theirscan|p, myscan, 0x4, 0x0;"
			"@p add.f32     myscan, theirscan, myscan;"
			"   shfl.up.b32 theirscan|p, myscan, 0x8, 0x0;"
			"@p add.f32     myscan, theirscan, myscan;"
			"   shfl.up.b32 theirscan|p, myscan, 0x10, 0x0;"
			"@p add.f32     myscan, theirscan, myscan;"
			"   mov.b32     %0, myscan;"
			"}"	: "+r"(myscan) );

		if (tid < 15) { // TODO: use PTX for compare
			asm("st.volatile.shared.u32 [%0+4], %1;" :: "r"( xmad( tid, 8.f, pshare ) ), "r"( xsub( myscan, mycount ) ) : "memory" );
			//* was: start_n_scan[wid][tid].y = myscan - mycount;
		}

		uint x13, y13, y14; // TODO: LDS.128
		asm("ld.volatile.shared.v2.u32 {%0,%1}, [%3+104];" // 104 = 13 x 8-byte uint2
			"ld.volatile.shared.u32     %2,     [%3+116];" // 116 = 14 x 8-bute uint2 + .y
			: "=r"(x13), "=r"(y13), "=r"(y14) : "r"(pshare) : "memory" );
		const uint dststart = x13;
		const uint lastdst  = xsub( xadd( dststart, y14 ), y13 );
		const uint nsrc     = y14;
		const uint spidext  = x13;
		//* was: const uint dststart = start_n_scan[wid][13].x;
		//* was: const uint lastdst  = xsub( xadd( dststart, start_n_scan[wid][14].y ), start_n_scan[wid][13].y );
		//* was: const uint nsrc     = start_n_scan[wid][14].y;
		//* was: const uint spidext  = start_n_scan[wid][13].x;

		uint nb = 0;
		for(uint p = 0; p < nsrc; p = xadd( p, 32u ) ) { // TODO: bool type PTX return
			const uint pid = p + tid;
			#ifdef LETS_MAKE_IT_MESSY
			uint spid;
			asm( "{ .reg .pred p, q, r;" // TODO: HOW TO USE LDS.128
				 "  .reg .f32  key;"
				 "  .reg .f32  scan3, scan6, scan9;"
				 "  .reg .f32  mystart, myscan;"
				 "  .reg .s32  array;"
				 "  .reg .f32  array_f;"
				 "   mov.b32           array, %4;"
				 "   ld.shared.f32     scan9,  [array +  9*8 + 4];"
				 "   setp.ge.f32       p, %1, scan9;"
				 "   selp.f32          key, %2, 0.0, p;"
				 "   mov.b32           array_f, array;"
				 "   fma.f32.rm        array_f, key, 8.0, array_f;"
				 "   mov.b32 array,    array_f;"
				 "   ld.shared.f32     scan3, [array + 3*8 + 4];"
				 "   setp.ge.f32       p, %1, scan3;"
				 "@p add.f32           key, key, %3;"
				 "   setp.lt.f32       p, key, %2;"
				 "   setp.lt.and.f32   p, %5, %6, p;"
				 "@p ld.shared.f32     scan6, [array + 6*8 + 4];"
				 "   setp.ge.and.f32   q, %1, scan6, p;"
				 "@q add.f32           key, key, %3;"
				 "   fma.f32.rm        array_f, key, 8.0, %4;"
				 "   mov.b32           array, array_f;"
				 "   ld.shared.v2.f32 {mystart, myscan}, [array];"
				 "   add.f32           mystart, mystart, %1;"
				 "   sub.f32           mystart, mystart, myscan;"
				 "   mov.b32           %0, mystart;"
				 "}" : "=r"(spid) : "f"(u2f(pid)), "f"(u2f(9u)), "f"(u2f(3u)), "f"(u2f(pshare)), "f"(u2f(pid)), "f"(u2f(nsrc)) );
			#else
			const uint key9 = 9*(pid >= start_n_scan[wid][9].y);
			uint key3 = 3*(pid >= start_n_scan[wid][key9 + 3].y);
			key3 += (key9 < 9) ? 3*(pid >= start_n_scan[wid][key9 + 6].y) : 0;
			const uint spid = pid - start_n_scan[wid][key3+key9].y + start_n_scan[wid][key3+key9].x;
			#endif

			const float4 xsrc = tex1Dfetch(texParticlesH4, xmin( spid, lastdst ) );

			for(uint dpid = dststart; dpid < lastdst; dpid = xadd(dpid, 1u) ) {
				const float4 xdest = tex1Dfetch( texParticlesH4, dpid );
				const float d2 = (xdest.x-xsrc.x)*(xdest.x-xsrc.x) + (xdest.y-xsrc.y)*(xdest.y-xsrc.y) + (xdest.z-xsrc.z)*(xdest.z-xsrc.z);

				#ifdef LETS_MAKE_IT_MESSY
				//int interacting = 0;
				asm volatile(".reg .pred interacting;");
				uint overview;
				asm("   setp.lt.ftz.f32  interacting, %3, 1.0;"
					"   setp.ne.and.f32  interacting, %1, %2, interacting;"
					"   setp.lt.and.f32  interacting, %2, %5, interacting;"
					"   vote.ballot.b32  %0, interacting;" :
					"=r"(overview) : "f"(u2f(dpid)), "f"(u2f(spid)), "f"(d2), "f"(u2f(1u)), "f"(u2f(lastdst)) );

				const uint insert = xadd( nb, i2u( __popc( overview & __lanemask_lt() ) ) );

				asm("@interacting st.volatile.shared.u32 [%0+1024], %1;" : :
					"r"( xmad( insert, 4.f, pshare ) ),
					"r"( __pack_8_24( xsub(dpid,dststart), spid ) ) :
					"memory" );
				//* was: if (interacting) queue[wid][insert] = __pack_8_24( xsub(dpid,dststart), spid );

				nb = xadd( nb, i2u( __popc( overview ) ) );
				if ( nb >= 32u ) {
					core_ytang( dststart, pshare, tid, spidext );
					nb = xsub( nb, 32u );

					asm("{ .reg .u32 tmp;"
						"   ld.volatile.shared.u32 tmp, [%0+1024+128];"
						"   st.volatile.shared.u32 [%0+1024], tmp;"
						"}" :: "r"( xmad( tid, 4.f, pshare ) ) : "memory" );
					//* was: queue[tid] = queue[tid+32];
				}
				#else
				const int interacting = (pid<nsrc) && ((dpid != spid) && (d2 < 1.0f));

				uint overview = __ballot( interacting );
				const uint insert = xadd( nb, i2u( __popc( overview & __lanemask_lt() ) ) );
				if (interacting) {
					asm("st.volatile.shared.u32 [%0+1024], %1;" : :
						"r"( xmad( insert, 4.f, pshare ) ),
						"r"( __pack_8_24( xsub(dpid,dststart), spid ) ) :
						"memory" );
					//* was: queue[insert] = __pack_8_24( xsub(dpid,dststart), spid );
				}
				nb = xadd( nb, i2u( __popc( overview ) ) );
				if ( nb >= 32u ) {
					core_ytang( dststart, pshare, tid, spidext );
					nb = xsub( nb, 32u );

					asm("{ .reg .u32 tmp;"
						"   ld.volatile.shared.u32 tmp, [%0+1024+128];"
						"   st.volatile.shared.u32 [%0+1024], tmp;"
						"}" :: "r"( xmad( tid, 4.f, pshare ) ) : "memory" );
					//* was: queue[tid] = queue[tid+32];
				}
				#endif

			}
		}

		if (tid < nb) {
			core_ytang( dststart, pshare, tid, spidext );
		}
		nb = 0;
	}
}

bool fdpd_init = false;

#include "../hacks.h"
#ifdef _TIME_PROFILE_
static cudaEvent_t evstart, evstop;
#endif

__global__ void make_texture( float4 *xyzouvwo, ushort4 *xyzo_half, const float *xyzuvw, const int n ) {
	for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=blockDim.x*gridDim.x) {
		float x = xyzuvw[i*6+0];
		float y = xyzuvw[i*6+1];
		float z = xyzuvw[i*6+2];
		float u = xyzuvw[i*6+3];
		float v = xyzuvw[i*6+4];
		float w = xyzuvw[i*6+5];
		xyzouvwo[i*2+0] = make_float4( x, y, z, 0.f );
		xyzouvwo[i*2+1] = make_float4( u, v, w, 0.f );
		xyzo_half[i] = make_ushort4( __float2half_rn(x), __float2half_rn(y), __float2half_rn(z), 0 );
	}
}

__global__ void make_texture2( uint2 *start_and_count, const int *start, const int *count, const int n ) {
	for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=blockDim.x*gridDim.x) {
		start_and_count[i] = make_uint2( start[i], count[i] );
	}
}

__global__ void check_acc(const int np)
{
	double sx = 0, sy = 0, sz = 0;
	for(int i=0;i<np;i++) {
		sx += info.axayaz[i*3+0];
		sy += info.axayaz[i*3+1];
		sz += info.axayaz[i*3+2];
	}
	printf("ACC: %lf %lf %lf\n",sx,sy,sz);
}

__global__ void transpose_acc(const int np)
{
	for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<np;i+=blockDim.x*gridDim.x) {
		int base = i & 0xFFFFFFE0U;
		int off  = i - base;
		float ax = info.axayaz[ base*3 + off      ];
		float ay = info.axayaz[ base*3 + off + 32 ];
		float az = info.axayaz[ base*3 + off + 64 ];
		// make sync between lanes
		if (__ballot(1)) {
			info.axayaz[ i * 3 + 0 ] = ax;
			info.axayaz[ i * 3 + 1 ] = ay;
			info.axayaz[ i * 3 + 2 ] = az;
		}
	}
}

void forces_dpd_cuda_nohost(const float * const xyzuvw, float * const axayaz,  const int np,
			    const int * const cellsstart, const int * const cellscount, 
			    const float rc,
			    const float XL, const float YL, const float ZL,
			    const float aij,
			    const float gamma,
			    const float sigma,
			    const float invsqrtdt,
			    const float seed, cudaStream_t stream)
{
//	#ifdef ONESTEP
//	cudaDeviceSetLimit( cudaLimitPrintfFifoSize, 32 * 1024 * 1024 );
//	#endif

	if (np == 0)
    {
	printf("WARNING: forces_dpd_cuda_nohost called with np = %d\n", np);
	return;
    }

    int nx = (int)ceil(XL / rc);
    int ny = (int)ceil(YL / rc);
    int nz = (int)ceil(ZL / rc);
    const int ncells = nx * ny * nz;

    if (!fdpd_init)
    {
	texStartAndCount.channelDesc = cudaCreateChannelDesc<uint2>();
	texStartAndCount.filterMode  = cudaFilterModePoint;
	texStartAndCount.mipmapFilterMode = cudaFilterModePoint;
	texStartAndCount.normalized = 0;

	texParticlesF4.channelDesc = cudaCreateChannelDesc<float4>();
	texParticlesF4.filterMode = cudaFilterModePoint;
	texParticlesF4.mipmapFilterMode = cudaFilterModePoint;
	texParticlesF4.normalized = 0;

	texParticlesH4.channelDesc = cudaCreateChannelDescHalf4();
	texParticlesH4.filterMode = cudaFilterModePoint;
	texParticlesH4.mipmapFilterMode = cudaFilterModePoint;
	texParticlesH4.normalized = 0;

	CUDA_CHECK(cudaFuncSetCacheConfig(_dpd_forces_symm_merged, cudaFuncCachePreferEqual));

#ifdef _TIME_PROFILE_
	CUDA_CHECK(cudaEventCreate(&evstart));
	CUDA_CHECK(cudaEventCreate(&evstop));
#endif
	fdpd_init = true;
    }

    InfoDPD c;

    size_t textureoffset;
	static  float4 *xyzouvwo;
	static ushort4 *xyzo_half;
	static int last_size;
	if (!xyzouvwo || last_size < np ) {
			if (xyzouvwo) {
				cudaFree( xyzouvwo );
				cudaFree( xyzo_half );
			}
			cudaMalloc( &xyzouvwo,  sizeof(float4)*2*np);
			cudaMalloc( &xyzo_half, sizeof(ushort4)*np);
			last_size = np;
	}
	static uint2 *start_and_count;
	static int last_nc;
	if (!start_and_count || last_nc < ncells) {
		if (start_and_count) {
			cudaFree( start_and_count );
		}
		cudaMalloc( &start_and_count, sizeof(uint2)*ncells);
		last_nc = ncells;
	}

	make_texture<<<64,512,0,stream>>>( xyzouvwo, xyzo_half, xyzuvw, np );
	CUDA_CHECK( cudaBindTexture( &textureoffset, &texParticlesF4, xyzouvwo,  &texParticlesF4.channelDesc, sizeof( float ) * 8 * np ) );
	assert(textureoffset == 0);
	CUDA_CHECK( cudaBindTexture( &textureoffset, &texParticlesH4, xyzo_half, &texParticlesH4.channelDesc, sizeof( ushort4 ) * np ) );
	assert(textureoffset == 0);
	make_texture2<<<64,512,0,stream>>>( start_and_count, cellsstart, cellscount, ncells );
    CUDA_CHECK(cudaBindTexture(&textureoffset, &texStartAndCount, start_and_count, &texStartAndCount.channelDesc, sizeof(uint2) * ncells));
    assert(textureoffset == 0);
      
    c.ncells = make_int3(nx, ny, nz);
    c.domainsize = make_float3(XL, YL, ZL);
    c.invdomainsize = make_float3(1 / XL, 1 / YL, 1 / ZL);
    c.domainstart = make_float3(-XL * 0.5, -YL * 0.5, -ZL * 0.5);
    c.invrc = 1.f / rc;
    c.aij = aij;
    c.gamma = gamma;
    c.sigmaf = sigma * invsqrtdt;
    c.axayaz = axayaz;
    c.seed = seed;
      
    CUDA_CHECK(cudaMemcpyToSymbolAsync(info, &c, sizeof(c), 0, cudaMemcpyHostToDevice, stream));
   
    static int cetriolo = 0;
    cetriolo++;

#ifdef _TIME_PROFILE_
    if (cetriolo % 500 == 0)
	CUDA_CHECK(cudaEventRecord(evstart));
#endif

    // YUHANG: fixed bug: not using stream
    CUDA_CHECK( cudaMemsetAsync(axayaz, 0, sizeof(float)*np*3, stream) );

    if (c.ncells.x%MYCPBX==0 && c.ncells.y%MYCPBY==0 && c.ncells.z%MYCPBZ==0) {
    	_dpd_forces_symm_merged<<<dim3(c.ncells.x/MYCPBX, c.ncells.y/MYCPBY, c.ncells.z/MYCPBZ), dim3(32, MYWPB), 0, stream>>>();
		#ifdef TRANSPOSED_ATOMICS
        transpose_acc<<<64,512,0,stream>>>(np);
		#endif
    }
    else {
    	fprintf(stderr,"Incompatible grid config\n");
    }

#ifdef ONESTEP
    check_acc<<<1,1>>>(np);
    CUDA_CHECK( cudaDeviceSynchronize() );
    CUDA_CHECK( cudaDeviceReset() );
    exit(0);
#endif

#ifdef _TIME_PROFILE_
    if (cetriolo % 500 == 0)
    {
	CUDA_CHECK(cudaEventRecord(evstop));
	CUDA_CHECK(cudaEventSynchronize(evstop));
	
	float tms;
	CUDA_CHECK(cudaEventElapsedTime(&tms, evstart, evstop));
	printf("elapsed time for DPD-BULK kernel: %.2f ms\n", tms);
    }
#endif

    CUDA_CHECK(cudaPeekAtLastError());	
}
