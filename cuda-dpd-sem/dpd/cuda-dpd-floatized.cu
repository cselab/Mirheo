/*
 *  cuda-dpd.cu
 *  Part of CTC/cuda-dpd-sem/dpd/
 *
 *  Created and authored by Yu-Hang Tang on 2015-03-18.
 *  Copyright 2015. All rights reserved.
 *
 *  Users are NOT authorized
 *  to employ the present software for their own publications
 *  before getting a written permission from the author of this file.
 */

#include <cstdio>
#include <cassert>

#include "../tiny-float.h"
#include "../dpd-rng.h"
#include "../hacks.h"

//#define PROF_TRIGGER
#define USE_TEXOBJ 3

struct InfoDPD {
    int3 ncells;
    float ncell_x, ncell_y;
    float3 domainsize, invdomainsize, domainstart;
    float invrc, aij, gamma, sigmaf;
    float * axayaz;
    float seed;
	#if (USE_TEXOBJ&1)
    cudaTextureObject_t txoParticles2;
    cudaTextureObject_t txoStart, txoCount;
	#endif
};

__constant__ InfoDPD info;

#if !(USE_TEXOBJ&2)
texture<float2, cudaTextureType1D> texParticles2;
texture<uint, cudaTextureType1D> texStart, texCount;
#endif
#if (USE_TEXOBJ&1)
template<typename TYPE> struct texture_object {
	cudaTextureObject_t txo;
	cudaResourceDesc res_desc;
	cudaTextureDesc tex_desc;
	TYPE *ptr_;
	long long n_;

	texture_object() : txo(0), ptr_(NULL), n_(0LL) {}

	inline operator cudaTextureObject_t () { return txo; };

	inline cudaTextureObject_t rebind( TYPE *ptr, const long long n ) {
		if ( ptr != ptr_ || ( ptr == ptr_ && n > n_ ) ) {
			if ( txo ) CUDA_CHECK( cudaDestroyTextureObject( txo ) );
			ptr_ = ptr;
			n_ = n;
			res_desc.resType = cudaResourceTypeLinear;
			res_desc.res.linear.desc = cudaCreateChannelDesc<TYPE>();
			res_desc.res.linear.devPtr = ptr_;
			res_desc.res.linear.sizeInBytes = sizeof( TYPE ) * n_;
			tex_desc.readMode = cudaReadModeElementType;
			CUDA_CHECK( cudaCreateTextureObject( &txo, &res_desc, &tex_desc, NULL ) );
		}
		return txo;
	}
};

texture_object<float2> txoParticles2;
texture_object<uint> txoStart, txoCount;
#endif

#define _XCPB_ 2
#define _YCPB_ 2
#define _ZCPB_ 1
#define CPB (_XCPB_ * _YCPB_ * _ZCPB_)
//#define  _TIME_PROFILE_
//#define _INSPECT_

#define LETS_MAKE_IT_MESSY

template<int s>
__device__ float viscosity_function( float x )
{
    return sqrtf( viscosity_function < s - 1 > ( x ) );
}

template<> __device__ float viscosity_function<0>( float x )
{
    return x;
}

// 88 FLOPS
__device__ float3 _dpd_interaction( const uint dpid, const float3 xdest, const float3 udest, const uint spid )
{
    const int sentry = xscale( spid, 3.f ); // 1 FLOP
	#if (USE_TEXOBJ&2)
    const float2 stmp0 = tex1Dfetch<float2>( info.txoParticles2, sentry           );
    const float2 stmp1 = tex1Dfetch<float2>( info.txoParticles2, xadd( sentry, 1 ) );
    const float2 stmp2 = tex1Dfetch<float2>( info.txoParticles2, xadd( sentry, 2 ) );
	#else
    const float2 stmp0 = tex1Dfetch( texParticles2, sentry           );
    const float2 stmp1 = tex1Dfetch( texParticles2, xadd( sentry, 1 ) ); // 1 FLOP
    const float2 stmp2 = tex1Dfetch( texParticles2, xadd( sentry, 2 ) ); // 1 FLOP
    #endif

    const float _xr = xdest.x - stmp0.x; // 1 FLOP
    const float _yr = xdest.y - stmp0.y; // 1 FLOP
    const float _zr = xdest.z - stmp1.x; // 1 FLOP

    const float rij2 = _xr * _xr + _yr * _yr + _zr * _zr; // 5 FLOPS
    assert( rij2 < 1.f );

    const float invrij = rsqrtf( rij2 ); // 1 FLOP
    const float rij = rij2 * invrij; // 1 FLOP
    const float wc = 1.f - rij; // 1 FLOP
    const float wr = viscosity_function < -VISCOSITY_S_LEVEL > ( wc ); // 0 FLOP

    const float xr = _xr * invrij; // 1 FLOP
    const float yr = _yr * invrij; // 1 FLOP
    const float zr = _zr * invrij; // 1 FLOP

    const float rdotv =
        xr * ( udest.x - stmp1.y ) +
        yr * ( udest.y - stmp2.x ) +
        zr * ( udest.z - stmp2.y );  // 8 FLOPS

    const float myrandnr = Logistic::mean0var1( info.seed, xmin(spid,dpid), xmax(spid,dpid) );  // 54+2 FLOP

    const float strength = info.aij * wc - ( info.gamma * wr * rdotv + info.sigmaf * myrandnr ) * wr; // 7 FLOPS

    return make_float3( strength * xr, strength * yr, strength * zr );
}

template<uint COLS, uint ROWS, uint NSRCMAX>
__device__ void core( const uint nsrc, const uint2 * const starts_and_scans, const uint p_starts_and_scans,
                      const uint ndst, const uint dststart )
{
    uint srcids[NSRCMAX];
    for( int i = 0; i < NSRCMAX; ++i )
        srcids[i] = 0;

    uint srccount = 0;
    assert( ndst == ROWS );

    const uint tid = threadIdx.x;
    const uint slot = tid / COLS;
    const uint subtid = tid % COLS;

    const uint dpid = xadd( dststart, slot ); // 1 FLOP
    const int entry = xscale( dpid, 3.f ); // 1 FLOP
	#if (USE_TEXOBJ&2)
    const float2 dtmp0 = tex1Dfetch<float2>( info.txoParticles2,       entry      );
    const float2 dtmp1 = tex1Dfetch<float2>( info.txoParticles2, xadd( entry, 1 ) );
    const float2 dtmp2 = tex1Dfetch<float2>( info.txoParticles2, xadd( entry, 2 ) );
	#else
    const float2 dtmp0 = tex1Dfetch( texParticles2,       entry      );
    const float2 dtmp1 = tex1Dfetch( texParticles2, xadd( entry, 1 ) ); // 1 FLOP
    const float2 dtmp2 = tex1Dfetch( texParticles2, xadd( entry, 2 ) ); // 1 FLOP
	#endif
    const float3 xdest = make_float3( dtmp0.x, dtmp0.y, dtmp1.x );
    const float3 udest = make_float3( dtmp1.y, dtmp2.x, dtmp2.y );

    float xforce = 0, yforce = 0, zforce = 0;

	for(uint s = 0; s < nsrc; s = xadd( s, COLS ) )
	{
		const uint pid  = xadd( s, subtid );
#ifdef LETS_MAKE_IT_MESSY
//		float key9f;
//		asm( "{ .reg .pred p, q;"
//			 "   setp.ge.f32 p, %1, %3;"
//			 "   setp.ge.f32 q, %1, %4;"
//			 "   selp.f32    %0, %2, 0.0, p;"
//			 "@q add.f32     %0, %0, %2; }" : "=f"(key9f) : "f"(u2f(pid)), "f"(u2f(9u)), "f"(u2f(starts_and_scans[9].y)), "f"(u2f(starts_and_scans[18].y)) );
//		const uint key9 = f2u(key9f);
//		asm( "{ .reg .pred p, q;"
//			 "   setp.ge.f32 p, %1, %3;"
//			 "   setp.ge.f32 q, %1, %4;"
//			 "@p add.f32     %0, %0, %2;"
//			 "@q add.f32     %0, %0, %2; }" : "+f"(key9f) : "f"(u2f(pid)), "f"(u2f(3u)), "f"(u2f(starts_and_scans[xadd(key9,3u)].y)), "f"(u2f(starts_and_scans[xadd(key9,6u)].y)) );
//		const uint key = f2u(key9f);

//		float key9f;
//		asm( "{ .reg .pred p, q;"
//			 "  .reg .f32  scan9, scan18;"
//			 "   ld.f32 scan9,  [%3 +  9*8 + 4];"
//			 "   ld.f32 scan18, [%3 + 18*8 + 4];"
//			 "   setp.ge.f32 p, %1, scan9;"
//			 "   setp.ge.f32 q, %1, scan18;"
//			 "   selp.f32    %0, %2, 0.0, p;"
//			 "@q add.f32     %0, %0, %2; }" : "=f"(key9f) : "f"(u2f(pid)), "f"(u2f(9u)), "l"(starts_and_scans) );
//		const uint key9 = f2u(key9f);
//		asm( "{ .reg .pred p, q;"
//			 "  .reg .f32  scan3, scan6;"
//			 "   ld.f32 scan3, [%3 + 3*8 + 4];"
//			 "   ld.f32 scan6, [%3 + 6*8 + 4];"
//			 "   setp.ge.f32 p, %1, scan3;"
//			 "   setp.ge.f32 q, %1, scan6;"
//			 "@p add.f32     %0, %0, %2;"
//			 "@q add.f32     %0, %0, %2; }" : "+f"(key9f) : "f"(u2f(pid)), "f"(u2f(3u)), "l"(starts_and_scans+key9) );
//		const uint key = f2u(key9f);

		uint key;
		asm( "{ .reg .pred p, q;"
			 "  .reg .f32  key;"
			 "  .reg .f32  scan3, scan6, scan9, scan18;"
			 "  .reg .s32  array;"
			 "  .reg .f32  array_f;"
			 "   mov.b32 array, %4;"
			 "   ld.shared.f32 scan9,  [array +  9*8 + 4];"
			 "   ld.shared.f32 scan18, [array + 18*8 + 4];"
			 "   setp.ge.f32 p, %1, scan9;"
			 "   setp.ge.f32 q, %1, scan18;"
			 "   selp.f32    key, %2, 0.0, p;"
			 "@q add.f32     key, key, %2;"
			 "   mov.b32 array_f, array;"
			 "   fma.f32.rm array_f, key, 8.0, array_f;"
			 "   mov.b32 array, array_f;"
			 "   ld.shared.f32 scan3, [array + 3*8 + 4];"
			 "   ld.shared.f32 scan6, [array + 6*8 + 4];"
			 "   setp.ge.f32 p, %1, scan3;"
			 "   setp.ge.f32 q, %1, scan6;"
			 "@p add.f32     key, key, %3;"
			 "@q add.f32     key, key, %3;"
	         "   mov.b32     %0, key;"
	         "}" : "=r"(key) : "f"(u2f(pid)), "f"(u2f(9u)), "f"(u2f(3u)), "r"(p_starts_and_scans) );
#else
		const uint key9 = xadd( xsel_ge( pid, scan9                , 9u, 0u ), xsel_ge( pid, scan18               , 9u, 0u ) );
		const uint key3 = xadd( xsel_ge( pid, scan[ xadd(key9,3u) ], 3u, 0u ), xsel_ge( pid, scan[ xadd(key9,6u) ], 3u, 0u ) );
		const uint key  = xadd( key9, key3 );
#endif

		const uint spid = xsub( xadd( pid, starts_and_scans[key].x ), starts_and_scans[key].y );

		#if (USE_TEXOBJ&2)
		const int sentry = xscale( spid, 3.f );
		const float2 stmp0 = tex1Dfetch<float2>( info.txoParticles2,       sentry      );
		const float2 stmp1 = tex1Dfetch<float2>( info.txoParticles2, xadd( sentry, 1 ) );
		#else
		const int sentry = xscale( spid, 3.f );
		const float2 stmp0 = tex1Dfetch( texParticles2,       sentry      );
		const float2 stmp1 = tex1Dfetch( texParticles2, xadd( sentry, 1 ) );
		#endif

		const float xdiff = xdest.x - stmp0.x;
		const float ydiff = xdest.y - stmp0.y;
		const float zdiff = xdest.z - stmp1.x;
#ifdef LETS_MAKE_IT_MESSY
		float srccount_f = u2f(srccount);
		asm("{ .reg .pred p;"
			"   setp.lt.f32 p, %1, %2;"
			"   setp.lt.and.f32 p, %3, 1.0, p;"
			"   setp.ne.and.f32 p, %4, %5, p;"
			"   @p st.u32 [%6], %7;"
			"   @p add.f32 %0, %0, %8;"
			"   }" : "+f"(srccount_f) : "f"(u2f(pid)), "f"(u2f(nsrc)), "f"(xdiff * xdiff + ydiff * ydiff + zdiff * zdiff), "f"(u2f(dpid)), "f"(u2f(spid)),
			"l"(srcids+srccount), "r"(spid), "f"(u2f(1u))
			: "memory" );
		srccount = f2u( srccount_f );
#else
		const float interacting = xfcmp_lt(pid, nsrc )
				                * xfcmp_lt( xdiff * xdiff + ydiff * ydiff + zdiff * zdiff, 1.f )
				                * xfcmp_ne( dpid, spid ) ;
		if (interacting) {
			srcids[srccount] = spid;
			srccount = xadd( srccount, 1u );
		}
#endif

		if ( srccount == NSRCMAX ) {
			srccount = xsub( srccount, 1u ); // 1 FLOP
			const float3 f = _dpd_interaction( dpid, xdest, udest, srcids[srccount] ); // 88 FLOPS

			xforce += f.x; // 1 FLOP
			yforce += f.y; // 1 FLOP
			zforce += f.z; // 1 FLOP
		}

		// 1 FLOP for s++
	}

#pragma unroll 4
    for( uint i = 0; i < srccount; i = xadd( i, 1u ) ) {
        const float3 f = _dpd_interaction( dpid, xdest, udest, srcids[i] ); // 88 FLOPS

        xforce += f.x; // 1 FLOP
        yforce += f.y; // 1 FLOP
        zforce += f.z; // 1 FLOP

        // 1 FLOP for i++
    }

    for( uint L = COLS / 2; L > 0; L >>= 1 ) {
        xforce += __shfl_xor( xforce, L ); // 1 FLOP
        yforce += __shfl_xor( yforce, L ); // 1 FLOP
        zforce += __shfl_xor( zforce, L ); // 1 FLOP
    }

    const float fcontrib = xsel_eq( subtid, 0u, xforce, xsel_eq( subtid, 1u, yforce, zforce ) ); // 2 FLOPS

    if( subtid < 3.f )
        info.axayaz[ xmad( dpid, 3.f, subtid ) ] = fcontrib;  // 2 FLOPS
}

template<uint COLS, uint ROWS, uint NSRCMAX>
__device__ void core_ilp( const uint nsrc, const uint2 * const starts_and_scans,
                          const uint ndst, const uint dststart )
{
    const uint tid    = threadIdx.x;
    const uint slot   = tid / COLS;
    const uint subtid = tid % COLS;

    const uint dpid = xadd( dststart, slot ); // 1 FLOP
    const int entry = xscale( dpid, 3.f ); // 1 FLOP
	#if (USE_TEXOBJ&2)
	const float2 dtmp0 = tex1Dfetch<float2>( info.txoParticles2,       entry      );
	const float2 dtmp1 = tex1Dfetch<float2>( info.txoParticles2, xadd( entry, 1 ) );
	const float2 dtmp2 = tex1Dfetch<float2>( info.txoParticles2, xadd( entry, 2 ) );
	#else
	const float2 dtmp0 = tex1Dfetch( texParticles2,       entry      );
	const float2 dtmp1 = tex1Dfetch( texParticles2, xadd( entry, 1 ) ); // 1 FLOP
	const float2 dtmp2 = tex1Dfetch( texParticles2, xadd( entry, 2 ) ); // 1 FLOP
	#endif
    const float3 xdest = make_float3( dtmp0.x, dtmp0.y, dtmp1.x );
    const float3 udest = make_float3( dtmp1.y, dtmp2.x, dtmp2.y );

    float xforce = 0, yforce = 0, zforce = 0;

    for( uint s = 0; s < nsrc; s = xadd( s, NSRCMAX * COLS ) ) {
        uint spids[NSRCMAX];
		#pragma unroll
        for( uint i = 0; i < NSRCMAX; ++i ) {
            const uint pid  = xadd( s, xmad( i, float(COLS), subtid ) );
#ifdef LETS_MAKE_IT_MESSY
			float key9f;
			asm( "{ .reg .pred p, q;"
				 "   setp.ge.f32 p, %1, %3;"
				 "   setp.ge.f32 q, %1, %4;"
				 "   selp.f32    %0, %2, 0.0, p;"
				 "@q add.f32     %0, %0, %2; }" : "=f"(key9f) : "f"(u2f(pid)), "f"(u2f(9u)), "f"(u2f(starts_and_scans[9].y)), "f"(u2f(starts_and_scans[18].y)) );
			const uint key9 = f2u(key9f);
			asm( "{ .reg .pred p, q;"
				 "   setp.ge.f32 p, %1, %3;"
				 "   setp.ge.f32 q, %1, %4;"
				 "@p add.f32     %0, %0, %2;"
				 "@q add.f32     %0, %0, %2; }" : "+f"(key9f) : "f"(u2f(pid)), "f"(u2f(3u)), "f"(u2f(starts_and_scans[xadd(key9,3u)].y)), "f"(u2f(starts_and_scans[xadd(key9,6u)].y)) );
			const uint key = f2u(key9f);
#else
    		const uint key9 = xadd( xsel_ge( pid, scan[ 9             ], 9u, 0u ), xsel_ge( pid, scan[ 18            ], 9u, 0u ) );
    		const uint key3 = xadd( xsel_ge( pid, scan[ xadd(key9,3u) ], 3u, 0u ), xsel_ge( pid, scan[ xadd(key9,6u) ], 3u, 0u ) );
    		const uint key  = xadd( key9, key3 );
#endif
            spids[i] = xsub( xadd( pid, starts_and_scans[key].x ), starts_and_scans[key].y );
        }

        uint interacting[NSRCMAX];
		#pragma unroll
        for( uint i = 0; i < NSRCMAX; ++i ) {
            const int sentry = xscale( spids[i], 3.f ); // 1 FLOP
			#if (USE_TEXOBJ&2)
			const float2 stmp0 = tex1Dfetch<float2>( info.txoParticles2,       sentry      );
			const float2 stmp1 = tex1Dfetch<float2>( info.txoParticles2, xadd( sentry, 1 ) );
			#else
			const float2 stmp0 = tex1Dfetch( texParticles2,       sentry      );
			const float2 stmp1 = tex1Dfetch( texParticles2, xadd( sentry, 1 ) ); // 1 FLOP
			#endif

            const float xdiff = xdest.x - stmp0.x;
            const float ydiff = xdest.y - stmp0.y;
            const float zdiff = xdest.z - stmp1.x;
#ifdef LETS_MAKE_IT_MESSY
			uint interacting_one;
            asm("{ .reg .pred p;"
				"   setp.lt.f32 p, %1, %2;"
				"   setp.lt.and.f32 p, %3, 1.0, p;"
				"   set.ne.and.u32.f32 %0, %4, %5, p;"
				"   }" : "=r"(interacting_one)  : "f"(u2f(xadd( s, xmad( i, float(COLS), subtid ) ))), "f"(u2f(nsrc)), "f"(xdiff * xdiff + ydiff * ydiff + zdiff * zdiff), "f"(u2f(dpid)), "f"(u2f(spids[i])) );
            interacting[i] = interacting_one;
#else
            interacting[i] = xfcmp_lt( xadd( s, xmad( i, float(COLS), subtid ) ), nsrc )
            		       * xfcmp_lt( xdiff * xdiff + ydiff * ydiff + zdiff * zdiff, 1.f )
            		       * xfcmp_ne( dpid, spids[i] );
#endif
        }

		#pragma unroll
        for( uint i = 0; i < NSRCMAX; ++i ) {
            if( interacting[i] ) {
                const float3 f = _dpd_interaction( dpid, xdest, udest, spids[i] ); // 88 FLOPS

                xforce += f.x; // 1 FLOP
                yforce += f.y; // 1 FLOP
                zforce += f.z; // 1 FLOP
            }
        }

        // 1 FLOP for s += NSRCMAX * COLS;
    }

    for( uint L = COLS / 2; L > 0; L >>= 1 ) {
        xforce += __shfl_xor( xforce, L ); // 1 FLOP
        yforce += __shfl_xor( yforce, L ); // 1 FLOP
        zforce += __shfl_xor( zforce, L ); // 1 FLOP
    }

    const float fcontrib = xsel_eq( subtid, 0u, xforce, xsel_eq( subtid, 1u, yforce, zforce ) );  // 2 FLOPS

    if( subtid < 3u )
        info.axayaz[ xmad( dpid, 3.f, subtid ) ] = fcontrib;  // 2 FLOPS
}

__global__ __launch_bounds__( 32 * CPB, 16 )
void _dpd_forces_floatized()
{
    assert( blockDim.x == warpSize && blockDim.y == CPB && blockDim.z == 1 );

    const uint tid = threadIdx.x;
    const uint wid = threadIdx.y;

    __shared__ volatile uint2 starts_and_scans[CPB][32];

    uint mycount = 0, myscan = 0;
    const int dx = ( tid ) % 3;
    const int dy = ( ( tid / 3 ) ) % 3;
    const int dz = ( ( tid / 9 ) ) % 3;

    if( tid < 27 ) {

        int xcid = blockIdx.x * _XCPB_ + ( ( threadIdx.y ) % _XCPB_ ) + dx - 1;
        int ycid = blockIdx.y * _YCPB_ + ( ( threadIdx.y / _XCPB_ ) % _YCPB_ ) + dy - 1;
        int zcid = blockIdx.z * _ZCPB_ + ( ( threadIdx.y / ( _XCPB_ * _YCPB_ ) ) % _ZCPB_ ) + dz - 1;

        const bool valid_cid =
                ( xcid >= 0 ) && ( xcid < info.ncells.x ) &&
                ( ycid >= 0 ) && ( ycid < info.ncells.y ) &&
                ( zcid >= 0 ) && ( zcid < info.ncells.z );

        xcid = xmin( xsub( info.ncells.x, 1 ), max( 0, xcid ) ); // 2 FLOPS
        ycid = xmin( xsub( info.ncells.y, 1 ), max( 0, ycid ) ); // 2 FLOPS
        zcid = xmin( xsub( info.ncells.z, 1 ), max( 0, zcid ) ); // 2 FLOPS

        const int cid = max( 0, ( zcid * info.ncells.y + ycid ) * info.ncells.x + xcid );
		#if (USE_TEXOBJ&2)
        starts_and_scans[wid][tid].x = tex1Dfetch<uint>( info.txoStart, cid );
        myscan = mycount = valid_cid ? tex1Dfetch<uint>( info.txoCount, cid ) : 0u;
		#else
        starts_and_scans[wid][tid].x = tex1Dfetch( texStart, cid );
        myscan = mycount = valid_cid ? tex1Dfetch( texCount, cid ) : 0u;
		#endif
    }

    for( uint L = 1u; L < 32u; L <<= 1 ) {
    	uint theirscan = i2u( __shfl_up( u2i(myscan), u2i(L) ) );
        myscan = xadd( myscan, xsel_ge( tid, L, theirscan, 0u ) ); // 2 FLOPS
    }

    if( tid < 28 )
    	starts_and_scans[wid][tid].y = xsub( myscan, mycount );

    const uint nsrc = starts_and_scans[wid][27].y;
    const uint dststart = starts_and_scans[wid][1 + 3 + 9].x;
    const uint ndst = xsub( starts_and_scans[wid][1 + 3 + 9 + 1].y, starts_and_scans[wid][1 + 3 + 9].y );
    const uint ndst4 = ( ndst >> 2 ) << 2;

    for( uint d = 0; d < ndst4; d = xadd( d, 4u ) )
        core<8, 4, 4>( nsrc, ( const uint2 * )starts_and_scans[wid], wid*32u*sizeof(uint2), 4, xadd( dststart, d ) );

    uint d = ndst4;
    if( xadd( d, 2u ) <= ndst ) {
        core<16, 2, 4>( nsrc, ( const uint2 * )starts_and_scans[wid], wid*32u*sizeof(uint2), 2, xadd( dststart, d ) );
        d = xadd( d, 2u );
    }

    if( d < ndst )
        core_ilp<32, 1, 2>( nsrc, ( const uint2 * )starts_and_scans[wid], 1, xadd( dststart, d ) );
}

#ifdef _COUNT_FLOPS
struct _dpd_interaction_flops_counter {
	const static unsigned long long FLOPS = 32ULL + Logistic::mean0var1_flops_counter::FLOPS;
};

template<uint COLS, uint ROWS, uint NSRCMAX>
__device__ void core_flops_counter( unsigned long long *FLOPS, const uint nsrc, const uint * const scan, const uint * const starts,
                      const uint ndst, const uint dststart )
{
    uint srcids[NSRCMAX];
    for( int i = 0; i < NSRCMAX; ++i )
        srcids[i] = 0;

    uint srccount = 0;
    assert( ndst == ROWS );

    const uint tid = threadIdx.x;
    const uint slot = tid / COLS;
    const uint subtid = tid % COLS;

    const uint dpid = xadd( dststart, slot ); // 1 FLOP
    const int entry = xscale( dpid, 3.f ); // 1 FLOP
	#if (USE_TEXOBJ&2)
    const float2 dtmp0 = tex1Dfetch<float2>( info.txoParticles2,       entry      );
    const float2 dtmp1 = tex1Dfetch<float2>( info.txoParticles2, xadd( entry, 1 ) );
    const float2 dtmp2 = tex1Dfetch<float2>( info.txoParticles2, xadd( entry, 2 ) );
	#else
    const float2 dtmp0 = tex1Dfetch( texParticles2,       entry      );
    const float2 dtmp1 = tex1Dfetch( texParticles2, xadd( entry, 1 ) ); // 1 FLOP
    const float2 dtmp2 = tex1Dfetch( texParticles2, xadd( entry, 2 ) ); // 1 FLOP
	#endif
    const float3 xdest = make_float3( dtmp0.x, dtmp0.y, dtmp1.x );
    const float3 udest = make_float3( dtmp1.y, dtmp2.x, dtmp2.y );

    atomicAdd( FLOPS, 4ULL );

    float xforce = 0, yforce = 0, zforce = 0;

	for(uint s = 0; s < nsrc; s = xadd( s, COLS ) )
	{
		const uint pid  = xadd( s, subtid );  // 1 FLOP
		const uint key9 = xadd( xsel_ge( pid, scan[ 9             ], 9u, 0u ), xsel_ge( pid, scan[ 18            ], 9u, 0u ) ); // 3 FLOPS
		const uint key3 = xadd( xsel_ge( pid, scan[ xadd(key9,3u) ], 3u, 0u ), xsel_ge( pid, scan[ xadd(key9,6u) ], 3u, 0u ) ); // 3 FLOPS
		const uint key  = xadd( key9, key3 ); // 1 FLOP

		const uint spid = xsub( xadd( pid, starts[key] ), scan[key] ); // 2 FLOPS

		const int sentry = xscale( spid, 3.f ); // 1 FLOP
		#if (USE_TEXOBJ&2)
		const float2 stmp0 = tex1Dfetch<float2>( info.txoParticles2,       sentry      );
		const float2 stmp1 = tex1Dfetch<float2>( info.txoParticles2, xadd( sentry, 1 ) );
		#else
		const float2 stmp0 = tex1Dfetch( texParticles2,       sentry      );
		const float2 stmp1 = tex1Dfetch( texParticles2, xadd( sentry, 1 ) ); // 1 FLOP
		#endif

		const float xdiff = xdest.x - stmp0.x; // 1 FLOP
		const float ydiff = xdest.y - stmp0.y; // 1 FLOP
		const float zdiff = xdest.z - stmp1.x; // 1 FLOP
		const float interacting = xfcmp_lt(pid, nsrc )
				                * xfcmp_lt( xdiff * xdiff + ydiff * ydiff + zdiff * zdiff, 1.f )
				                * xfcmp_ne( dpid, spid ) ; // 10 FLOPS

		atomicAdd( FLOPS, 25ULL );

		if (interacting) {
			srcids[srccount] = spid;
			srccount = xadd( srccount, 1u ); // 1 FLOP
			atomicAdd( FLOPS, 1ULL );
		}

		if( srccount == NSRCMAX ) {
			srccount = xsub( srccount, 1u ); // 1 FLOP
			//const float3 f = _dpd_interaction( dpid, xdest, udest, srcids[srccount] ); // 88 FLOPS
			//xforce += f.x; // 1 FLOP
			//yforce += f.y; // 1 FLOP
			//zforce += f.z; // 1 FLOP
			atomicAdd( FLOPS, _dpd_interaction_flops_counter::FLOPS + 4ULL );
		}

		// 1 FLOP for s++
		atomicAdd( FLOPS, 1ULL );
	}

#pragma unroll 4
    for( uint i = 0; i < srccount; i = xadd( i, 1u ) ) {
        // const float3 f = _dpd_interaction( dpid, xdest, udest, srcids[i] ); // 88 FLOPS

        // xforce += f.x; // 1 FLOP
        // yforce += f.y; // 1 FLOP
        // zforce += f.z; // 1 FLOP

        // 1 FLOP for i++
        atomicAdd( FLOPS, _dpd_interaction_flops_counter::FLOPS + 4ULL );
    }

    for( uint L = COLS / 2; L > 0; L >>= 1 ) {
        // xforce += __shfl_xor( xforce, L ); // 1 FLOP
        // yforce += __shfl_xor( yforce, L ); // 1 FLOP
        // zforce += __shfl_xor( zforce, L ); // 1 FLOP
        atomicAdd( FLOPS, 3ULL );
    }

    // const float fcontrib = xsel_eq( subtid, 0u, xforce, xsel_eq( subtid, 1u, yforce, zforce ) ); // 2 FLOPS
    atomicAdd( FLOPS, 2ULL );

    if( subtid < 3.f ) {
        // info.axayaz[ xmad( dpid, 3.f, subtid ) ] = fcontrib;  // 2 FLOPS
        atomicAdd( FLOPS, 2ULL );
    }
}

template<uint COLS, uint ROWS, uint NSRCMAX>
__device__ void core_ilp_flops_counter( unsigned long long *FLOPS, const uint nsrc, const uint * const scan, const uint * const starts,
                          const uint ndst, const uint dststart )
{
    const uint tid    = threadIdx.x;
    const uint slot   = tid / COLS;
    const uint subtid = tid % COLS;

    const uint dpid = xadd( dststart, slot ); // 1 FLOP
    const int entry = xscale( dpid, 3.f ); // 1 FLOP
	#if (USE_TEXOBJ&2)
	const float2 dtmp0 = tex1Dfetch<float2>( info.txoParticles2,       entry      );
	const float2 dtmp1 = tex1Dfetch<float2>( info.txoParticles2, xadd( entry, 1 ) );
	const float2 dtmp2 = tex1Dfetch<float2>( info.txoParticles2, xadd( entry, 2 ) );
	#else
	const float2 dtmp0 = tex1Dfetch( texParticles2,       entry      );
	const float2 dtmp1 = tex1Dfetch( texParticles2, xadd( entry, 1 ) ); // 1 FLOP
	const float2 dtmp2 = tex1Dfetch( texParticles2, xadd( entry, 2 ) ); // 1 FLOP
	#endif
    const float3 xdest = make_float3( dtmp0.x, dtmp0.y, dtmp1.x );
    const float3 udest = make_float3( dtmp1.y, dtmp2.x, dtmp2.y );

    atomicAdd( FLOPS, 4ULL );

    float xforce = 0, yforce = 0, zforce = 0;

    for( uint s = 0; s < nsrc; s = xadd( s, NSRCMAX * COLS ) ) {
        uint spids[NSRCMAX];
		#pragma unroll
        for( uint i = 0; i < NSRCMAX; ++i ) {
            const uint pid  = xadd( s, xmad( i, float(COLS), subtid ) ); // 3 FLOPS
    		const uint key9 = xadd( xsel_ge( pid, scan[ 9             ], 9u, 0u ), xsel_ge( pid, scan[ 18            ], 9u, 0u ) ); // 3 FLOPS
    		const uint key3 = xadd( xsel_ge( pid, scan[ xadd(key9,3u) ], 3u, 0u ), xsel_ge( pid, scan[ xadd(key9,6u) ], 3u, 0u ) ); // 3 FLOPS
    		const uint key  = xadd( key9, key3 );

            spids[i] = xsub( xadd( pid, starts[key] ), scan[key] ); // 2 FLOPS
            atomicAdd( FLOPS, 11ULL );
        }

        bool interacting[NSRCMAX];
		#pragma unroll
        for( uint i = 0; i < NSRCMAX; ++i ) {
            const int sentry = xscale( spids[i], 3.f ); // 1 FLOP
			#if (USE_TEXOBJ&2)
			const float2 stmp0 = tex1Dfetch<float2>( info.txoParticles2,       sentry      );
			const float2 stmp1 = tex1Dfetch<float2>( info.txoParticles2, xadd( sentry, 1 ) );
			#else
			const float2 stmp0 = tex1Dfetch( texParticles2,       sentry      );
			const float2 stmp1 = tex1Dfetch( texParticles2, xadd( sentry, 1 ) ); // 1 FLOP
			#endif

            const float xdiff = xdest.x - stmp0.x; // 1 FLOP
            const float ydiff = xdest.y - stmp0.y; // 1 FLOP
            const float zdiff = xdest.z - stmp1.x; // 1 FLOP
            interacting[i] = xfcmp_lt( xadd( s, xmad( i, float(COLS), subtid ) ), nsrc )
            		       * xfcmp_lt( xdiff * xdiff + ydiff * ydiff + zdiff * zdiff, 1.f )
            		       * xfcmp_ne( dpid, spids[i] ); // 13 FLOPS
            atomicAdd( FLOPS, 18ULL );
        }

		#pragma unroll
        for( uint i = 0; i < NSRCMAX; ++i ) {
            if( interacting[i] ) {
                //const float3 f = _dpd_interaction( dpid, xdest, udest, spids[i] ); // 88 FLOPS

                //xforce += f.x; // 1 FLOP
                //yforce += f.y; // 1 FLOP
                //zforce += f.z; // 1 FLOP
            	atomicAdd( FLOPS, _dpd_interaction_flops_counter::FLOPS + 3ULL );
            }
        }

        // 1 FLOP for s += NSRCMAX * COLS;
    }

    for( uint L = COLS / 2; L > 0; L >>= 1 ) {
        //xforce += __shfl_xor( xforce, L ); // 1 FLOP
        //yforce += __shfl_xor( yforce, L ); // 1 FLOP
        //zforce += __shfl_xor( zforce, L ); // 1 FLOP
    	atomicAdd( FLOPS, 3ULL );
    }

    //const float fcontrib = xsel_eq( subtid, 0u, xforce, xsel_eq( subtid, 1u, yforce, zforce ) );  // 2 FLOPS
    atomicAdd( FLOPS, 2ULL );

    if( subtid < 3u ) {
        //info.axayaz[ xmad( dpid, 3.f, subtid ) ] = fcontrib;  // 2 FLOPS
    	atomicAdd( FLOPS, 2ULL );
    }
}

__global__ __launch_bounds__( 32 * CPB, 16 )
void _dpd_forces_floatized_flops_counter(unsigned long long *FLOPS)
{
    assert( blockDim.x == warpSize && blockDim.y == CPB && blockDim.z == 1 );

    const uint tid = threadIdx.x;
    const uint wid = threadIdx.y;

    __shared__ volatile uint starts[CPB][32], scan[CPB][32];

    uint mycount = 0, myscan = 0;
    if( tid < 27 ) {
        const int dx = ( tid ) % 3;
        const int dy = ( ( tid / 3 ) ) % 3;
        const int dz = ( ( tid / 9 ) ) % 3;

        int xcid = blockIdx.x * _XCPB_ + ( ( threadIdx.y ) % _XCPB_ ) + dx - 1;
        int ycid = blockIdx.y * _YCPB_ + ( ( threadIdx.y / _XCPB_ ) % _YCPB_ ) + dy - 1;
        int zcid = blockIdx.z * _ZCPB_ + ( ( threadIdx.y / ( _XCPB_ * _YCPB_ ) ) % _ZCPB_ ) + dz - 1;

        const bool valid_cid =
                ( xcid >= 0 ) && ( xcid < info.ncells.x ) &&
                ( ycid >= 0 ) && ( ycid < info.ncells.y ) &&
                ( zcid >= 0 ) && ( zcid < info.ncells.z );

        xcid = xmin( xsub( info.ncells.x, 1 ), max( 0, xcid ) ); // 2 FLOPS
        ycid = xmin( xsub( info.ncells.y, 1 ), max( 0, ycid ) ); // 2 FLOPS
        zcid = xmin( xsub( info.ncells.z, 1 ), max( 0, zcid ) ); // 2 FLOPS
        atomicAdd( FLOPS, 6ULL );

        const int cid = max( 0, ( zcid * info.ncells.y + ycid ) * info.ncells.x + xcid );
		#if (USE_TEXOBJ&2)
        starts[wid][tid] = tex1Dfetch<uint>( info.txoStart, cid );
        myscan = mycount = valid_cid ? tex1Dfetch<uint>( info.txoCount, cid ) : 0u;
		#else
        starts[wid][tid] = tex1Dfetch( texStart, cid );
        myscan = mycount = valid_cid ? tex1Dfetch( texCount, cid ) : 0u;
		#endif
    }

    for( uint L = 1u; L < 32u; L <<= 1 ) {
    	uint theirscan = i2u( __shfl_up( u2i(myscan), u2i(L) ) );
        myscan = xadd( myscan, xsel_ge( tid, L, theirscan, 0u ) ); // 2 FLOPS
        atomicAdd( FLOPS, 2ULL );
    }

    if( tid < 28 ) {
        scan[wid][tid] = xsub( myscan, mycount ); // 1 FLOP
        atomicAdd( FLOPS, 2ULL );
    }

    const uint nsrc = scan[wid][27];
    const uint dststart = starts[wid][1 + 3 + 9];
    const uint ndst = xsub( scan[wid][1 + 3 + 9 + 1], scan[wid][1 + 3 + 9] ); // 1 FLOP
    atomicAdd( FLOPS, 1ULL );
    const uint ndst4 = ( ndst >> 2 ) << 2;

    for( uint d = 0; d < ndst4; d = xadd( d, 4u ) ) { // 1 FLOP
        core_flops_counter<8, 4, 4>( FLOPS, nsrc, ( const uint * )scan[wid], ( const uint * )starts[wid], 4, xadd( dststart, d ) ); // 1 FLOP
    	atomicAdd( FLOPS, 2ULL );
	}

    uint d = ndst4;
    if( xadd( d, 2u ) <= ndst ) { // 1 FLOP
        core_flops_counter<16, 2, 4>( FLOPS, nsrc, ( const uint * )scan[wid], ( const uint * )starts[wid], 2, xadd( dststart, d ) ); // 1 FLOP
        d = xadd( d, 2u ); // 1 FLOP
        atomicAdd( FLOPS, 3ULL );
    }

    if( d < ndst ) {
        core_ilp_flops_counter<32, 1, 2>( FLOPS, nsrc, ( const uint * )scan[wid], ( const uint * )starts[wid], 1, xadd( dststart, d ) ); // 1 FLOP
        atomicAdd( FLOPS, 1ULL );
    }
}

__global__ void reset_flops( unsigned long long *FLOPS ) {
	*FLOPS = 0ULL;
}

__global__ void print_flops( unsigned long long *FLOPS ) {
	printf("FLOPS count: %llu\n", *FLOPS);
}
#endif

bool fdpd_init = false;

#include "../hacks.h"
#ifdef _TIME_PROFILE_
static cudaEvent_t evstart, evstop;
#endif

void forces_dpd_cuda_nohost( const float * const xyzuvw, float * const axayaz,  const int np,
                             const int * const cellsstart, const int * const cellscount,
                             const float rc,
                             const float XL, const float YL, const float ZL,
                             const float aij,
                             const float gamma,
                             const float sigma,
                             const float invsqrtdt,
                             const float seed, cudaStream_t stream )
{
	if( np == 0 ) {
        printf( "WARNING: forces_dpd_cuda_nohost called with np = %d\n", np );
        return;
    }

    int nx = ( int )ceil( XL / rc );
    int ny = ( int )ceil( YL / rc );
    int nz = ( int )ceil( ZL / rc );
    const int ncells = nx * ny * nz;

	#if !(USE_TEXOBJ&2)
    size_t textureoffset;
    CUDA_CHECK( cudaBindTexture( &textureoffset, &texParticles2, xyzuvw, &texParticles2.channelDesc, sizeof( float ) * 6 * np ) );
    assert( textureoffset == 0 );
    CUDA_CHECK( cudaBindTexture( &textureoffset, &texStart, cellsstart, &texStart.channelDesc, sizeof( uint ) * ncells ) );
    assert( textureoffset == 0 );
    CUDA_CHECK( cudaBindTexture( &textureoffset, &texCount, cellscount, &texCount.channelDesc, sizeof( uint ) * ncells ) );
    assert( textureoffset == 0 );
	#endif

    InfoDPD c;
    c.ncells = make_int3( nx, ny, nz );
    c.ncell_x = nx;
    c.ncell_y = ny;
    c.domainsize = make_float3( XL, YL, ZL );
    c.invdomainsize = make_float3( 1 / XL, 1 / YL, 1 / ZL );
    c.domainstart = make_float3( -XL * 0.5, -YL * 0.5, -ZL * 0.5 );
    c.invrc = 1.f / rc;
    c.aij = aij;
    c.gamma = gamma;
    c.sigmaf = sigma * invsqrtdt;
    c.axayaz = axayaz;
    c.seed = seed;
	#if (USE_TEXOBJ&1)
    c.txoParticles2 = txoParticles2.rebind( (float2*)const_cast<float*>(xyzuvw), 3 * np );
    c.txoStart = txoStart.rebind( (uint*)const_cast<int*>(cellsstart), ncells );
    c.txoCount = txoCount.rebind( (uint*)const_cast<int*>(cellscount), ncells );
	#endif

	if( !fdpd_init ) {
		#if !(USE_TEXOBJ&2)
        texStart.channelDesc = cudaCreateChannelDesc<uint>();
        texStart.filterMode = cudaFilterModePoint;
        texStart.mipmapFilterMode = cudaFilterModePoint;
        texStart.normalized = 0;

        texCount.channelDesc = cudaCreateChannelDesc<uint>();
        texCount.filterMode = cudaFilterModePoint;
        texCount.mipmapFilterMode = cudaFilterModePoint;
        texCount.normalized = 0;

        texParticles2.channelDesc = cudaCreateChannelDesc<float2>();
        texParticles2.filterMode = cudaFilterModePoint;
        texParticles2.mipmapFilterMode = cudaFilterModePoint;
        texParticles2.normalized = 0;
		#endif

	void ( *dpdkernel )() =  _dpd_forces_floatized;

        CUDA_CHECK( cudaFuncSetCacheConfig( *dpdkernel, cudaFuncCachePreferL1 ) );

#ifdef _TIME_PROFILE_
        CUDA_CHECK( cudaEventCreate( &evstart ) );
        CUDA_CHECK( cudaEventCreate( &evstop ) );
#endif
        fdpd_init = true;
    }

    CUDA_CHECK( cudaMemcpyToSymbolAsync( info, &c, sizeof( c ), 0, cudaMemcpyHostToDevice, stream ) );

    static int cetriolo = 0;
    cetriolo++;

#ifdef _TIME_PROFILE_
    if( cetriolo % 500 == 0 )
        CUDA_CHECK( cudaEventRecord( evstart ) );
#endif
    _dpd_forces_floatized <<< dim3( c.ncells.x / _XCPB_,
                          c.ncells.y / _YCPB_,
                          c.ncells.z / _ZCPB_ ), dim3( 32, CPB ), 0, stream >>> ();

#ifdef _COUNT_FLOPS
    {
    	static unsigned long long *FLOPS;
    	if (!FLOPS) cudaMalloc( &FLOPS, 128 * sizeof(unsigned long long) );
    	reset_flops<<<1,1,0,stream>>>(FLOPS);
    	_dpd_forces_floatized_flops_counter <<< dim3( c.ncells.x / _XCPB_,
    	                          c.ncells.y / _YCPB_,
    	                          c.ncells.z / _ZCPB_ ), dim3( 32, CPB ), 0, stream >>> ( FLOPS );
    	print_flops<<<1,1,0,stream>>>(FLOPS);

    	//count FLOPS
        //report data to scree

//        if( cetriolo % 1000 == 0 ) {
//            enum { COLS = 16, ROWS = 2 };
//
//            const size_t nentries = np * COLS;
//
//            int2 * data;
//            CUDA_CHECK( cudaHostAlloc( &data, sizeof( int2 ) * nentries, cudaHostAllocMapped ) );
//            memset( data, 0xff, sizeof( int2 ) * nentries );
//
//            int * devptr;
//            CUDA_CHECK( cudaHostGetDevicePointer( &devptr, data, 0 ) );
//
//            inspect_dpd_forces <<< dim3( c.ncells.x / _XCPB_, c.ncells.y / _YCPB_, c.ncells.z / _ZCPB_ ), dim3( 32, CPB ), 0, stream >>>
//            ( COLS, ROWS, np, data, nentries );
//
//            CUDA_CHECK( cudaDeviceSynchronize() );
//
//            char path2report[2000];
//            sprintf( path2report, "inspection-%d-tstep.txt", cetriolo );
//
//            FILE * f = fopen( path2report, "w" );
//            assert( f );
//
//            for( int i = 0, c = 0; i < np; ++i ) {
//                fprintf( f, "pid %05d: ", i );
//
//                int s = 0, pot = 0;
//                for( int j = 0; j < COLS; ++j, ++c ) {
//                    fprintf( f, "%02d ", data[c].x );
//                    s += data[c].x;
//                    pot += data[c].y;
//                }
//
//                fprintf( f, " sum: %02d pot: %d\n", s, ( pot + COLS - 1 ) / ( COLS ) );
//            }
//
//            fclose( f );
//
//            CUDA_CHECK( cudaFreeHost( data ) );
//            printf( "inspection saved to %s.\n", path2report );
//        }
    }
#endif

#ifdef _TIME_PROFILE_
    if( cetriolo % 500 == 0 ) {
        CUDA_CHECK( cudaEventRecord( evstop ) );
        CUDA_CHECK( cudaEventSynchronize( evstop ) );

        float tms;
        CUDA_CHECK( cudaEventElapsedTime( &tms, evstart, evstop ) );
        printf( "elapsed time for DPD-BULK kernel: %.2f ms\n", tms );
    }
#endif

    CUDA_CHECK( cudaPeekAtLastError() );
}

#include <cmath>
#include <unistd.h>

#include "../cell-lists.h"

int fdpd_oldnp = 0, fdpd_oldnc = 0;

float * fdpd_xyzuvw = NULL, * fdpd_axayaz = NULL;
int * fdpd_start = NULL, * fdpd_count = NULL;

void forces_dpd_cuda_aos( float * const _xyzuvw, float * const _axayaz,
                          int * const order, const int np,
                          const float rc,
                          const float XL, const float YL, const float ZL,
                          const float aij,
                          const float gamma,
                          const float sigma,
                          const float invsqrtdt,
                          const float seed,
                          const bool nohost )
{
    if( np == 0 ) {
        printf( "WARNING: forces_dpd_cuda_aos called with np = %d\n", np );
        return;
    }

    int nx = ( int )ceil( XL / rc );
    int ny = ( int )ceil( YL / rc );
    int nz = ( int )ceil( ZL / rc );
    const int ncells = nx * ny * nz;

    if( !fdpd_init ) {
		#if !(USE_TEXOBJ&2)
        texStart.channelDesc = cudaCreateChannelDesc<uint>();
        texStart.filterMode = cudaFilterModePoint;
        texStart.mipmapFilterMode = cudaFilterModePoint;
        texStart.normalized = 0;

        texCount.channelDesc = cudaCreateChannelDesc<uint>();
        texCount.filterMode = cudaFilterModePoint;
        texCount.mipmapFilterMode = cudaFilterModePoint;
        texCount.normalized = 0;

        texParticles2.channelDesc = cudaCreateChannelDesc<float2>();
        texParticles2.filterMode = cudaFilterModePoint;
        texParticles2.mipmapFilterMode = cudaFilterModePoint;
        texParticles2.normalized = 0;
		#endif

        fdpd_init = true;
    }

    if( fdpd_oldnp < np ) {
        if( fdpd_oldnp > 0 ) {
            CUDA_CHECK( cudaFree( fdpd_xyzuvw ) );
            CUDA_CHECK( cudaFree( fdpd_axayaz ) );
        }

        CUDA_CHECK( cudaMalloc( &fdpd_xyzuvw, sizeof( float ) * 6 * np ) );
        CUDA_CHECK( cudaMalloc( &fdpd_axayaz, sizeof( float ) * 3 * np ) );

		#if !(USE_TEXOBJ&2)
        size_t textureoffset;
        CUDA_CHECK( cudaBindTexture( &textureoffset, &texParticles2, fdpd_xyzuvw, &texParticles2.channelDesc, sizeof( float ) * 6 * np ) );
		#endif

        fdpd_oldnp = np;
    }

    if( fdpd_oldnc < ncells ) {
        if( fdpd_oldnc > 0 ) {
            CUDA_CHECK( cudaFree( fdpd_start ) );
            CUDA_CHECK( cudaFree( fdpd_count ) );
        }

        CUDA_CHECK( cudaMalloc( &fdpd_start, sizeof( uint ) * ncells ) );
        CUDA_CHECK( cudaMalloc( &fdpd_count, sizeof( uint ) * ncells ) );

		#if !(USE_TEXOBJ&2)
        size_t textureoffset = 0;
        CUDA_CHECK( cudaBindTexture( &textureoffset, &texStart, fdpd_start, &texStart.channelDesc, sizeof( uint ) * ncells ) );
        CUDA_CHECK( cudaBindTexture( &textureoffset, &texCount, fdpd_count, &texCount.channelDesc, sizeof( uint ) * ncells ) );
		#endif

        fdpd_oldnc = ncells;
    }

    CUDA_CHECK( cudaMemcpyAsync( fdpd_xyzuvw, _xyzuvw, sizeof( float ) * np * 6, nohost ? cudaMemcpyDeviceToDevice : cudaMemcpyHostToDevice, 0 ) );

    InfoDPD c;
    c.ncells = make_int3( nx, ny, nz );
    c.ncell_x = nx;
    c.ncell_y = ny;
    c.domainsize = make_float3( XL, YL, ZL );
    c.invdomainsize = make_float3( 1 / XL, 1 / YL, 1 / ZL );
    c.domainstart = make_float3( -XL * 0.5, -YL * 0.5, -ZL * 0.5 );
    c.invrc = 1.f / rc;
    c.aij = aij;
    c.gamma = gamma;
    c.sigmaf = sigma * invsqrtdt;
    c.axayaz = fdpd_axayaz;
    c.seed = seed;

    build_clists( fdpd_xyzuvw, np, rc, c.ncells.x, c.ncells.y, c.ncells.z,
                  c.domainstart.x, c.domainstart.y, c.domainstart.z,
                  order, fdpd_start, fdpd_count, NULL );

    //TextureWrap texStart(_ptr(starts), ncells), texCount(_ptr(counts), ncells);
    //TextureWrap texParticles((float2*)_ptr(xyzuvw), 3 * np);

    CUDA_CHECK( cudaMemcpyToSymbolAsync( info, &c, sizeof( c ), 0 ) );

    _dpd_forces_floatized <<< dim3( c.ncells.x / _XCPB_,
                          c.ncells.y / _YCPB_,
                          c.ncells.z / _ZCPB_ ), dim3( 32, CPB ) >>> ();

    CUDA_CHECK( cudaPeekAtLastError() );

//copy xyzuvw as well?!?
    if( nohost ) {
        CUDA_CHECK( cudaMemcpy( _xyzuvw, fdpd_xyzuvw, sizeof( float ) * 6 * np, cudaMemcpyDeviceToDevice ) );
        CUDA_CHECK( cudaMemcpy( _axayaz, fdpd_axayaz, sizeof( float ) * 3 * np, cudaMemcpyDeviceToDevice ) );
    } else
        CUDA_CHECK( cudaMemcpy( _axayaz, fdpd_axayaz, sizeof( float ) * 3 * np, cudaMemcpyDeviceToHost ) );

#ifdef _CHECK_
    CUDA_CHECK( cudaThreadSynchronize() );

    for( int ii = 0; ii < np; ++ii ) {
        printf( "pid %d -> %f %f %f\n", ii, ( float )axayaz[0 + 3 * ii], ( float )axayaz[1 + 3 * ii], ( float )axayaz[2 + 3 * ii] );

        int cnt = 0;
        float fc = 0;
        const int i = order[ii];
        printf( "devi coords are %f %f %f\n", ( float )xyzuvw[0 + 6 * ii], ( float )xyzuvw[1 + 6 * ii], ( float )xyzuvw[2 + 6 * ii] );
        printf( "host coords are %f %f %f\n", ( float )_xyzuvw[0 + 6 * i], ( float )_xyzuvw[1 + 6 * i], ( float )_xyzuvw[2 + 6 * i] );

        for( int j = 0; j < np; ++j ) {
            if( i == j )
                continue;

            float xr = _xyzuvw[0 + 6 * i] - _xyzuvw[0 + 6 * j];
            float yr = _xyzuvw[1 + 6 * i] - _xyzuvw[1 + 6 * j];
            float zr = _xyzuvw[2 + 6 * i] - _xyzuvw[2 + 6 * j];

            xr -= c.domainsize.x *  ::floor( 0.5f + xr / c.domainsize.x );
            yr -= c.domainsize.y *  ::floor( 0.5f + yr / c.domainsize.y );
            zr -= c.domainsize.z *  ::floor( 0.5f + zr / c.domainsize.z );

            const float rij2 = xr * xr + yr * yr + zr * zr;
            const float invrij = rsqrtf( rij2 );
            const float rij = rij2 * invrij;
            const float wr = max( ( float )0, 1 - rij * c.invrc );

            const bool collision =  rij2 < 1;

            if( collision )
                fc += wr;// printf("ref p %d colliding with %d\n", i, j);

            cnt += collision;
        }
        printf( "i found %d host interactions and with cuda i found %d\n", cnt, ( int )axayaz[0 + 3 * ii] );
        assert( cnt == ( float )axayaz[0 + 3 * ii] );
        printf( "fc aij ref %f vs cuda %e\n", fc, ( float )axayaz[1 + 3 * ii] );
        assert( fabs( fc - ( float )axayaz[1 + 3 * ii] ) < 1e-4 );
    }

    printf( "test done.\n" );
    sleep( 1 );
    exit( 0 );
#endif
}


int * fdpd_order = NULL;
float * fdpd_pv = NULL, *fdpd_a = NULL;

void forces_dpd_cuda( const float * const xp, const float * const yp, const float * const zp,
                      const float * const xv, const float * const yv, const float * const zv,
                      float * const xa, float * const ya, float * const za,
                      const int np,
                      const float rc,
                      const float LX, const float LY, const float LZ,
                      const float aij,
                      const float gamma,
                      const float sigma,
                      const float invsqrtdt,
                      const float seed )
{
    if( np <= 0 ) return;

    if( np > fdpd_oldnp ) {
        if( fdpd_oldnp > 0 ) {
            CUDA_CHECK( cudaFreeHost( fdpd_pv ) );
            CUDA_CHECK( cudaFreeHost( fdpd_order ) );
            CUDA_CHECK( cudaFreeHost( fdpd_a ) );
        }

        CUDA_CHECK( cudaHostAlloc( &fdpd_pv, sizeof( float ) * np * 6, cudaHostAllocDefault ) );
        CUDA_CHECK( cudaHostAlloc( &fdpd_order, sizeof( int ) * np, cudaHostAllocDefault ) );
        CUDA_CHECK( cudaHostAlloc( &fdpd_a, sizeof( float ) * np * 3, cudaHostAllocDefault ) );

        //this will be done by forces_dpd_cuda
        //fdpd_oldnp = np;
    }

    for( int i = 0; i < np; ++i ) {
        fdpd_pv[0 + 6 * i] = xp[i];
        fdpd_pv[1 + 6 * i] = yp[i];
        fdpd_pv[2 + 6 * i] = zp[i];
        fdpd_pv[3 + 6 * i] = xv[i];
        fdpd_pv[4 + 6 * i] = yv[i];
        fdpd_pv[5 + 6 * i] = zv[i];
    }

    forces_dpd_cuda_aos( fdpd_pv, fdpd_a, fdpd_order, np, rc, LX, LY, LZ,
                         aij, gamma, sigma, invsqrtdt, seed, false );

    //delete [] pv;

    for( int i = 0; i < np; ++i ) {
        xa[fdpd_order[i]] += fdpd_a[0 + 3 * i];
        ya[fdpd_order[i]] += fdpd_a[1 + 3 * i];
        za[fdpd_order[i]] += fdpd_a[2 + 3 * i];
    }

    //delete [] a;

    //delete [] order;
}
