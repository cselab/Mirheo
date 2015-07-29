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
#include <mpi.h>

#include "cuda-dpd.h"
#include "../dpd-rng.h"

struct InfoDPD {
    int3 ncells;
    uint nxyz;
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
#define HALF_FLOAT

#define _XCPB_ 2
#define _YCPB_ 2
#define _ZCPB_ 1
#define CPB (_XCPB_ * _YCPB_ * _ZCPB_)

__device__ float3 _dpd_interaction( const int dpid, const float4 xdest, const float4 udest, const float4 xsrc, const float4 usrc, const int spid )
{
    const float _xr = xdest.x - xsrc.x;
    const float _yr = xdest.y - xsrc.y;
    const float _zr = xdest.z - xsrc.z;

    const float rij2 = _xr * _xr + _yr * _yr + _zr * _zr;
    assert( rij2 < 1 );

    const float invrij = rsqrtf( rij2 );
    const float rij = rij2 * invrij;
    const float wc = max(0.f, 1 - rij);
    const float wr = viscosity_function < -VISCOSITY_S_LEVEL > ( wc );

    const float xr = _xr * invrij;
    const float yr = _yr * invrij;
    const float zr = _zr * invrij;

    const float rdotv =
        xr * ( udest.x - usrc.x ) +
        yr * ( udest.y - usrc.y ) +
        zr * ( udest.z - usrc.z );

    const float myrandnr = Logistic::mean0var1( info.seed, xmin( spid, dpid ), xmax( spid, dpid ) );

    const float strength = info.aij * wc - ( info.gamma * wr * rdotv + info.sigmaf * myrandnr ) * wr;

    return make_float3( strength * xr, strength * yr, strength * zr );
}

#define __IMOD(x,y) ((x)-((x)/(y))*(y))

__inline__ __device__ uint __lanemask_lt()
{
    uint mask;
    asm( "mov.u32 %0, %lanemask_lt;" : "=r"( mask ) );
    return mask;
}

__inline__ __device__ uint __pack_8_24( uint a, uint b )
{
    uint d;
    asm( "bfi.b32  %0, %1, %2, 24, 8;" : "=r"( d ) : "r"( a ), "r"( b ) );
    return d;
}

__inline__ __device__ uint2 __unpack_8_24( uint d )
{
    uint a;
    asm( "bfe.u32  %0, %1, 24, 8;" : "=r"( a ) : "r"( d ) );
    return make_uint2( a, d & 0x00FFFFFFU );
}

__device__ char4 tid2ind[32] = {
    { -1, -1, -1, 0}, {0, -1, -1, 0}, {1, -1, -1, 0},
    { -1,  0, -1, 0}, {0,  0, -1, 0}, {1,  0, -1, 0},
    { -1 , 1, -1, 0}, {0,  1, -1, 0}, {1,  1, -1, 0},
    { -1, -1,  0, 0}, {0, -1,  0, 0}, {1, -1,  0, 0},
    { -1,  0,  0, 0}, {0,  0,  0, 0}, {1,  0,  0, 0},
    { -1,  1,  0, 0}, {0,  1,  0, 0}, {1,  1,  0, 0},
    { -1, -1,  1, 0}, {0, -1,  1, 0}, {1, -1,  1, 0},
    { -1,  0,  1, 0}, {0,  0,  1, 0}, {1,  0,  1, 0},
    { -1,  1,  1, 0}, {0,  1,  1, 0}, {1,  1,  1, 0},
    { 0,  0,  0, 0}, {0,  0,  0, 0}, {0,  0,  0, 0},
    { 0,  0,  0, 0}, {0,  0,  0, 0}
};

// TODO: modify compiled PTX to replace integer comparison in branch statements

__forceinline__ __device__ void core_ytang( const uint dststart, const uint pshare, const uint tid, const uint spidext )
{
    uint item;
    const uint offset = xmad( tid, 4.f, pshare );
    asm volatile( "ld.volatile.shared.u32 %0, [%1+1024];" : "=r"( item ) : "r"( offset ) : "memory" );
    const uint2 pid = __unpack_8_24( item );
    const uint dpid = xadd( dststart, pid.x );
    const uint spid = pid.y;

    const uint dentry = xscale( dpid, 2.f );
    const uint sentry = xscale( spid, 2.f );
    const float4 xdest = tex1Dfetch( texParticlesF4,       dentry );
    const float4 xsrc  = tex1Dfetch( texParticlesF4,       sentry );
    const float4 udest = tex1Dfetch( texParticlesF4, xadd( dentry, 1u ) );
    const float4 usrc  = tex1Dfetch( texParticlesF4, xadd( sentry, 1u ) );
    const float3 f = _dpd_interaction( dpid, xdest, udest, xsrc, usrc, spid );

    // the overhead of transposition acc back
    // can be completely killed by changing the integration kernel
#ifdef TRANSPOSED_ATOMICS
    uint off  = dpid & 0x0000001FU;
    uint base = xdiv( dpid, 1 / 32.f );
    float* acc = info.axayaz + xmad( base, 96.f, off );
    atomicAdd( acc   , f.x );
    atomicAdd( acc + 32, f.y );
    atomicAdd( acc + 64, f.z );

    if( spid < spidext ) {
        uint off  = spid & 0x0000001FU;
        uint base = xdiv( spid, 1 / 32.f );
        float* acc = info.axayaz + xmad( base, 96.f, off );
        atomicAdd( acc   , -f.x );
        atomicAdd( acc + 32, -f.y );
        atomicAdd( acc + 64, -f.z );
    }
#else
    float* acc = info.axayaz + dpid * 3;
    atomicAdd( acc    , f.x );
    atomicAdd( acc + 1, f.y );
    atomicAdd( acc + 2, f.z );

    if( spid < spidext ) {
        float* acc = info.axayaz + spid * 3;
        atomicAdd( acc    , -f.x );
        atomicAdd( acc + 1, -f.y );
        atomicAdd( acc + 2, -f.z );
    }
#endif
}

#define MYCPBX  (4)
#define MYCPBY  (2)
#define MYCPBZ  (2)
#define MYWPB   (4)

__global__  __launch_bounds__( 32 * MYWPB, 16 )
void _dpd_forces_symm_merged()
{

    asm volatile( ".shared .u32 smem[512];" ::: "memory" );
    //* was: __shared__ uint2 volatile start_n_scan[MYWPB][32];
    //* was: __shared__ uint  volatile queue[MYWPB][64];

    const uint tid = threadIdx.x;
    const uint wid = threadIdx.y;
    const uint pshare = xscale( threadIdx.y, 256.f );

#if !(defined(__CUDA_ARCH__)) || __CUDA_ARCH__>= 350
    const char4 offs = __ldg( tid2ind + tid );
#else
    const char4 offs = tid2ind[tid];
#endif

    const int cbase = blockIdx.z * MYCPBZ * info.ncells.x * info.ncells.y +
                      blockIdx.y * MYCPBY * info.ncells.x +
                      blockIdx.x * MYCPBX + wid +
                      offs.z * info.ncells.x * info.ncells.y +
                      offs.y * info.ncells.x +
                      offs.x;

    for( uint it = 0; it < 4 ; it = xadd( it, 1u ) ) {

        //* was: const int cid = cbase + ( it < 2u ) * info.ncells.x * info.ncells.y +
        //* was:                 ( ( it & 1u ) ^ ( it >> 1 ) ) * info.ncells.x;
        int cid;
        asm( "{  .reg .pred    p;"
             "   .reg .f32     incf;"
             "   .reg .s32     inc;"
             "    setp.lt.f32  p, %2, %3;"
             "    selp.f32     incf, %4, 0.0, p;"
             "    add.f32      incf, incf, %5;"
             "    mov.b32      inc, incf;"
             "    mul.lo.u32   inc, inc, %6;"
             "    add.s32 %0,  %1, inc;"
             "}" :
             "=r"( cid ) : "r"( cbase ), "f"( u2f( it ) ), "f"( u2f( 2u ) ), "f"( i2f( info.ncells.y ) ), "f"( u2f( ( it & 1u ) ^ ( it >> 1 ) ) ),
             "r"( info.ncells.x ) );

        //* was: uint mycount=0, myscan=0;
        //* was: if (tid < 14) {
        //* was: const int cid = cbase +
        //* was:        (it>1)*info.ncells.x*info.ncells.y +
        //* was:        ((it&1)^((it>>1)&1))*info.ncells.x;

        //* was: const bool valid_cid = (cid >= 0) && (cid < info.ncells.x*info.ncells.y*info.ncells.z);
        //* was: const uint2 sc = valid_cid ? tex1Dfetch( texStartAndCount, cid ) : make_uint2(0,0);

        //* was: start_n_scan[wid][tid].x = (valid_cid) ? sc.x : 0;
        //* was: myscan = mycount = (valid_cid) ? sc.y : 0;
        //* was: }
        uint mystart = 0, mycount = 0, myscan;
        asm( "{  .reg .pred vc;"
             "   .reg .u32  foo, bar;"
             "    setp.lt.f32     vc, %2, %3;"
             "    setp.ge.and.f32 vc, %5, 0.0, vc;"
             "    setp.lt.and.s32 vc, %4, %6, vc;"
             "    selp.s32 %0, 1, 0, vc;"
             "@vc tex.1d.v4.s32.s32 {%0, %1, foo, bar}, [texStartAndCount, %4];"
             "}" :
             "+r"( mystart ), "+r"( mycount )  :
             "f"( u2f( tid ) ), "f"( u2f( 14u ) ), "r"( cid ), "f"( i2f( cid ) ),
             "r"( info.nxyz ) );
        myscan  = mycount;
        asm volatile( "st.volatile.shared.u32 [%0], %1;" ::
                      "r"( xmad( tid, 8.f, pshare ) ),
                      "r"( mystart ) :
                      "memory" );

        // was: #pragma unroll
        // was: for(int L = 1; L < 32; L <<= 1) {
        // was:     myscan = xadd( myscan, (tid >= L)*__shfl_up(myscan, L) );
        // was: }
        asm( "{ .reg .pred   p;"
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
             "}" : "+r"( myscan ) );


        //* was: if (tid < 15) start_n_scan[wid][tid].y = myscan - mycount;
        asm volatile( "{    .reg .pred lt15;"
                      "      setp.lt.f32 lt15, %0, %1;"
                      "@lt15 st.volatile.shared.u32 [%2+4], %3;"
                      "}":: "f"( u2f( tid ) ), "f"( u2f( 15u ) ), "r"( xmad( tid, 8.f, pshare ) ), "r"( xsub( myscan, mycount ) ) : "memory" );

        //* was: const uint dststart = start_n_scan[wid][13].x;
        //* was: const uint lastdst  = xsub( xadd( dststart, start_n_scan[wid][14].y ), start_n_scan[wid][13].y );
        //* was: const uint nsrc     = start_n_scan[wid][14].y;
        //* was: const uint spidext  = start_n_scan[wid][13].x;
        uint x13, y13, y14; // TODO: LDS.128
        asm volatile( "ld.volatile.shared.v2.u32 {%0,%1}, [%3+104];" // 104 = 13 x 8-byte uint2
                      "ld.volatile.shared.u32     %2,     [%3+116];" // 116 = 14 x 8-bute uint2 + .y
                      : "=r"( x13 ), "=r"( y13 ), "=r"( y14 ) : "r"( pshare ) : "memory" );
        const uint dststart = x13;
        const uint lastdst  = xsub( xadd( dststart, y14 ), y13 );
        const uint nsrc     = y14;
        const uint spidext  = x13;

        uint nb = 0;

        for( uint p = 0; p < nsrc; p = xadd( p, 32u ) ) {

            const uint pid = p + tid;

            //* was: const uint key9 = 9*(pid >= start_n_scan[wid][9].y);
            //* was: uint key3 = 3*(pid >= start_n_scan[wid][key9 + 3].y);
            //* was: key3 += (key9 < 9) ? 3*(pid >= start_n_scan[wid][key9 + 6].y) : 0;
            //* was: const uint spid = pid - start_n_scan[wid][key3+key9].y + start_n_scan[wid][key3+key9].x;
            uint spid;
            asm volatile( "{ .reg .pred p, q, r;" // TODO: HOW TO USE LDS.128
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
                          "   ld.shared.f32     scan6, [array + 6*8 + 4];"
                          "   setp.ge.and.f32   q, %1, scan6, p;"
                          "@q add.f32           key, key, %3;"
                          "   fma.f32.rm        array_f, key, 8.0, %4;"
                          "   mov.b32           array, array_f;"
                          "   ld.shared.v2.f32 {mystart, myscan}, [array];"
                          "   add.f32           mystart, mystart, %1;"
                          "   sub.f32           mystart, mystart, myscan;"
                          "   mov.b32           %0, mystart;"
                          "}" : "=r"( spid ) : "f"( u2f( pid ) ), "f"( u2f( 9u ) ), "f"( u2f( 3u ) ), "f"( u2f( pshare ) ), "f"( u2f( pid ) ), "f"( u2f( nsrc ) ) );

            const float4 xsrc = tex1Dfetch( texParticlesH4, xmin( spid, lastdst ) );

            for( uint dpid = dststart; dpid < lastdst; dpid = xadd( dpid, 1u ) ) {

                const float4 xdest = tex1Dfetch( texParticlesH4, dpid );
                const float dx = xdest.x - xsrc.x;
                const float dy = xdest.y - xsrc.y;
                const float dz = xdest.z - xsrc.z;
                const float d2 = dx * dx + dy * dy + dz * dz;

                //* was: int interacting = 0;
                asm volatile( ".reg .pred interacting;" );
                uint overview;
                asm( "   setp.lt.ftz.f32  interacting, %3, 1.0;"
                     "   setp.ne.and.f32  interacting, %1, %2, interacting;"
                     "   setp.lt.and.f32  interacting, %2, %5, interacting;"
                     "   vote.ballot.b32  %0, interacting;" :
                     "=r"( overview ) : "f"( u2f( dpid ) ), "f"( u2f( spid ) ), "f"( d2 ), "f"( u2f( 1u ) ), "f"( u2f( lastdst ) ) );

                const uint insert = xadd( nb, i2u( __popc( overview & __lanemask_lt() ) ) );

                //* was: if (interacting) queue[wid][insert] = __pack_8_24( xsub(dpid,dststart), spid );
                asm volatile( "@interacting st.volatile.shared.u32 [%0+1024], %1;" : :
                              "r"( xmad( insert, 4.f, pshare ) ),
                              "r"( __pack_8_24( xsub( dpid, dststart ), spid ) ) :
                              "memory" );

                nb = xadd( nb, i2u( __popc( overview ) ) );
                if( nb >= 32u ) {
                    core_ytang( dststart, pshare, tid, spidext );
                    nb = xsub( nb, 32u );

                    //* was: queue[tid] = queue[tid+32];
                    asm volatile( "{ .reg .u32 tmp;"
                                  "   ld.volatile.shared.u32 tmp, [%0+1024+128];"
                                  "   st.volatile.shared.u32 [%0+1024], tmp;"
                                  "}" :: "r"( xmad( tid, 4.f, pshare ) ) : "memory" );
                }

            }
        }

        if( tid < nb ) {
            core_ytang( dststart, pshare, tid, spidext );
        }
        nb = 0;
    }
}

bool fdpd_init = false;
static bool is_mps_enabled = false;
#include "../hacks.h"
#ifdef _TIME_PROFILE_
static cudaEvent_t evstart, evstop;
#endif

__global__ void make_texture( float4 * __restrict xyzouvwo, ushort4 * __restrict xyzo_half, const float * __restrict xyzuvw, const uint n )
{
    extern __shared__ volatile float  smem[];
    const uint warpid = threadIdx.x / 32;
    const uint lane = threadIdx.x % 32;
    //for( uint i = ( blockIdx.x * blockDim.x + threadIdx.x ) & 0xFFFFFFE0U ; i < n ; i += blockDim.x * gridDim.x ) {
    const uint i =  (blockIdx.x * blockDim.x + threadIdx.x ) & 0xFFFFFFE0U;

    const float2 * base = ( float2* )( xyzuvw +  i * 6 );
#pragma unroll 3
        for( uint j = lane; j < 96; j += 32 ) {
            float2 u = base[j];
            // NVCC bug: no operator = between volatile float2 and float2
            asm volatile( "st.volatile.shared.v2.f32 [%0], {%1, %2};" : : "r"( ( warpid * 96 + j )*8 ), "f"( u.x ), "f"( u.y ) : "memory" );
        }
        // SMEM: XYZUVW XYZUVW ...
        uint pid = lane / 2;
        const uint x_or_v = ( lane % 2 ) * 3;
        xyzouvwo[ i * 2 + lane ] = make_float4( smem[ warpid * 192 + pid * 6 + x_or_v + 0 ],
                                                smem[ warpid * 192 + pid * 6 + x_or_v + 1 ],
                                                smem[ warpid * 192 + pid * 6 + x_or_v + 2 ], 0 );
        pid += 16;
        xyzouvwo[ i * 2 + lane + 32] = make_float4( smem[ warpid * 192 + pid * 6 + x_or_v + 0 ],
                                       smem[ warpid * 192 + pid * 6 + x_or_v + 1 ],
                                       smem[ warpid * 192 + pid * 6 + x_or_v + 2 ], 0 );

        xyzo_half[i + lane] = make_ushort4( __float2half_rn( smem[ warpid * 192 + lane * 6 + 0 ] ),
                                            __float2half_rn( smem[ warpid * 192 + lane * 6 + 1 ] ),
                                            __float2half_rn( smem[ warpid * 192 + lane * 6 + 2 ] ), 0 );
// }
}

__global__ void make_texture2( uint2 *start_and_count, const int *start, const int *count, const int n )
{
    for( int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x ) {
        start_and_count[i] = make_uint2( start[i], count[i] );
    }
}

__global__ void check_acc( const int np )
{
    double sx = 0, sy = 0, sz = 0;
    for( int i = 0; i < np; i++ ) {
        double ax = info.axayaz[i * 3 + 0];
        double ay = info.axayaz[i * 3 + 1];
        double az = info.axayaz[i * 3 + 2];
        if( ax != ax || ay != ay || az != az ) {
            printf( "particle %d: %f %f %f\n", i, ax, ay, az );
        }
        sx += ax;
        sy += ay;
        sz += az;
    }
    printf( "ACC: %+.7lf %+.7lf %+.7lf\n", sx, sy, sz );
}

__global__ void check_acc_transposed( const int np )
{
    double sx = 0, sy = 0, sz = 0;
    for( int i = 0; i < np; i++ ) {
        int base = i / 32;
        int off = i % 32;
        int p = base * 96 + off;
        sx += info.axayaz[p];
        sy += info.axayaz[p + 32];
        sz += info.axayaz[p + 64];
    }
    printf( "ACC-TRANSPOSED: %+.7lf %+.7lf %+.7lf\n", sx, sy, sz );
}


__global__  __launch_bounds__( 1024, 2 )
void transpose_acc( const int np )
{
    __shared__ volatile float  smem[32][96];
    const uint lane = threadIdx.x % 32;
    const uint warpid = threadIdx.x / 32;

    for( uint i = ( blockIdx.x * blockDim.x + threadIdx.x ) & 0xFFFFFFE0U; i < np; i += blockDim.x * gridDim.x ) {
        const uint base = xmad( i, 3.f, lane );
        smem[warpid][lane   ] = info.axayaz[ base      ];
        smem[warpid][lane + 32] = info.axayaz[ base + 32 ];
        smem[warpid][lane + 64] = info.axayaz[ base + 64 ];
        info.axayaz[ base      ] = smem[warpid][ xmad( __IMOD( lane + 0, 3 ), 32.f, ( lane + 0 ) / 3 ) ];
        info.axayaz[ base + 32 ] = smem[warpid][ xmad( __IMOD( lane + 32, 3 ), 32.f, ( lane + 32 ) / 3 ) ];
        info.axayaz[ base + 64 ] = smem[warpid][ xmad( __IMOD( lane + 64, 3 ), 32.f, ( lane + 64 ) / 3 ) ];
    }
}

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
//  #ifdef ONESTEP
//  cudaDeviceSetLimit( cudaLimitPrintfFifoSize, 32 * 1024 * 1024 );
//  #endif

    if( np == 0 ) {
        printf( "WARNING: forces_dpd_cuda_nohost called with np = %d\n", np );
        return;
    }

    int nx = ( int )ceil( XL / rc );
    int ny = ( int )ceil( YL / rc );
    int nz = ( int )ceil( ZL / rc );
    const int ncells = nx * ny * nz;

    if( !fdpd_init ) {
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

        CUDA_CHECK( cudaFuncSetCacheConfig( _dpd_forces_symm_merged, cudaFuncCachePreferEqual ) );
        CUDA_CHECK( cudaFuncSetCacheConfig( make_texture, cudaFuncCachePreferShared ) );

#ifdef _TIME_PROFILE_
        CUDA_CHECK( cudaEventCreate( &evstart ) );
        CUDA_CHECK( cudaEventCreate( &evstop ) );
#endif

	{
	    is_mps_enabled = false;

	    const char * mps_variables[] = {
		"CRAY_CUDA_MPS",
		"CUDA_MPS",
		"CRAY_CUDA_PROXY",
		"CUDA_PROXY"
	    };

	    for(int i = 0; i < 4; ++i)
		is_mps_enabled |= getenv(mps_variables[i])!= NULL && atoi(getenv(mps_variables[i])) != 0;
	}

        fdpd_init = true;
    }

    static InfoDPD c;

    size_t textureoffset;
    static  float  *xyzouvwo;
    static ushort4 *xyzo_half;
    static int last_size;
    if( !xyzouvwo || last_size < np ) {
        if( xyzouvwo ) {
            cudaFree( xyzouvwo );
            cudaFree( xyzo_half );
        }
        last_size = np;
        if( last_size % 32 ) last_size += 32 - last_size % 32;
        cudaMalloc( &xyzouvwo,  sizeof( float4 ) * 2 * last_size );
        cudaMalloc( &xyzo_half, sizeof( ushort4 ) * last_size );
    }
    static uint2 *start_and_count;
    static int last_nc;
    if( !start_and_count || last_nc < ncells ) {
        if( start_and_count ) {
            cudaFree( start_and_count );
        }
        cudaMalloc( &start_and_count, sizeof( uint2 )*ncells );
        last_nc = ncells;
    }

    make_texture <<< (np + 1023)/1024, 1024, 1024 * 6 * sizeof( float ), stream>>>( ( float4* )xyzouvwo, xyzo_half, xyzuvw, np );
    CUDA_CHECK( cudaBindTexture( &textureoffset, &texParticlesF4, xyzouvwo,  &texParticlesF4.channelDesc, sizeof( float ) * 8 * np ) );
    assert( textureoffset == 0 );
    CUDA_CHECK( cudaBindTexture( &textureoffset, &texParticlesH4, xyzo_half, &texParticlesH4.channelDesc, sizeof( ushort4 ) * np ) );
    assert( textureoffset == 0 );
    make_texture2 <<< 64, 512, 0, stream>>>( start_and_count, cellsstart, cellscount, ncells );
    CUDA_CHECK( cudaBindTexture( &textureoffset, &texStartAndCount, start_and_count, &texStartAndCount.channelDesc, sizeof( uint2 ) * ncells ) );
    assert( textureoffset == 0 );

    c.ncells = make_int3( nx, ny, nz );
    c.nxyz = nx * ny * nz;
    c.domainsize = make_float3( XL, YL, ZL );
    c.invdomainsize = make_float3( 1 / XL, 1 / YL, 1 / ZL );
    c.domainstart = make_float3( -XL * 0.5, -YL * 0.5, -ZL * 0.5 );
    c.invrc = 1.f / rc;
    c.aij = aij;
    c.gamma = gamma;
    c.sigmaf = sigma * invsqrtdt;
    c.axayaz = axayaz;
    c.seed = seed;

    if (!is_mps_enabled)
	CUDA_CHECK( cudaMemcpyToSymbolAsync( info, &c, sizeof( c ), 0, cudaMemcpyHostToDevice, stream ) );
    else
	CUDA_CHECK( cudaMemcpyToSymbol( info, &c, sizeof( c ), 0, cudaMemcpyHostToDevice ) );

    static int cetriolo = 0;
    cetriolo++;

#ifdef _TIME_PROFILE_
    if( cetriolo % 500 == 0 )
        CUDA_CHECK( cudaEventRecord( evstart ) );
#endif

    // YUHANG: fixed bug: not using stream
    int np32 = np;
    if( np32 % 32 ) np32 += 32 - np32 % 32;
    CUDA_CHECK( cudaMemsetAsync( axayaz, 0, sizeof( float )* np32 * 3, stream ) );

    if( c.ncells.x % MYCPBX == 0 && c.ncells.y % MYCPBY == 0 && c.ncells.z % MYCPBZ == 0 ) {
        _dpd_forces_symm_merged <<< dim3( c.ncells.x / MYCPBX, c.ncells.y / MYCPBY, c.ncells.z / MYCPBZ ), dim3( 32, MYWPB ), 0, stream >>> ();
#ifdef TRANSPOSED_ATOMICS
        // check_acc_transposed<<<1, 1, 0, stream>>>( np );
        transpose_acc <<< 28, 1024, 0, stream>>>( np );
#endif
    } else {
        fprintf( stderr, "Incompatible grid config\n" );
    }

#ifdef ONESTEP
    check_acc <<< 1, 1, 0, stream>>>( np );
    CUDA_CHECK( cudaDeviceSynchronize() );
    CUDA_CHECK( cudaDeviceReset() );
    MPI_Finalize();
    exit( 0 );
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
