/*
 *  cuda-dpd-bipartite.cu
 *  Part of CTC/cuda-dpd-sem/dpd/
 *
 *  Created and authored by Yu-Hang Tang on 2015-03-18.
 *  Copyright 2015. All rights reserved.
 *
 *  Users are NOT authorized
 *  to employ the present software for their own publications
 *  before getting a written permission from the author of this file.
 */

#include <cassert>

#include "../dpd-rng.h"

struct BipartiteInfoDPD {
    int3 ncells;
    float3 domainsize, invdomainsize, domainstart;
    float invrc, aij, gamma, sigmaf;
};

__constant__ BipartiteInfoDPD bipart_info;

#ifndef NDEBUG
//#define _CHECK_
#endif

const static uint COLS = 8;
const static uint ROWS = (32 / COLS);
const static uint CPB = 4;

#include "../hacks.h"

__global__
void _bipartite_dpd_directforces( float * const axayaz, const int np, const int np_src,
                                  const float seed, const int mask, const float * xyzuvw, const float * xyzuvw_src,
                                  const float invrc, const float aij, const float gamma, const float sigmaf )
{
    assert( blockDim.x % warpSize == 0 );
    assert( blockDim.x * gridDim.x >= np );

    const int tid = threadIdx.x % warpSize;
    const int pid = threadIdx.x + blockDim.x * blockIdx.x;
    const bool valid = pid < np;

    float xp, yp, zp, up, vp, wp;

    if( valid ) {
        xp = xyzuvw[0 + pid * 6];
        yp = xyzuvw[1 + pid * 6];
        zp = xyzuvw[2 + pid * 6];
        up = xyzuvw[3 + pid * 6];
        vp = xyzuvw[4 + pid * 6];
        wp = xyzuvw[5 + pid * 6];
    }

    float xforce = 0, yforce = 0, zforce = 0;

    for( int s = 0; s < np_src; s += warpSize ) {
        float my_xq, my_yq, my_zq, my_uq, my_vq, my_wq;

        const int batchsize = min( warpSize, np_src - s );

        if( tid < batchsize ) {
            my_xq = xyzuvw_src[0 + ( tid + s ) * 6];
            my_yq = xyzuvw_src[1 + ( tid + s ) * 6];
            my_zq = xyzuvw_src[2 + ( tid + s ) * 6];
            my_uq = xyzuvw_src[3 + ( tid + s ) * 6];
            my_vq = xyzuvw_src[4 + ( tid + s ) * 6];
            my_wq = xyzuvw_src[5 + ( tid + s ) * 6];
        }

        for( int l = 0; l < batchsize; ++l ) {
            const float xq = __shfl( my_xq, l );
            const float yq = __shfl( my_yq, l );
            const float zq = __shfl( my_zq, l );
            const float uq = __shfl( my_uq, l );
            const float vq = __shfl( my_vq, l );
            const float wq = __shfl( my_wq, l );

            //necessary to force the execution shuffles here below
            //__syncthreads();

            //if (valid)
            {
                const float _xr = xp - xq;
                const float _yr = yp - yq;
                const float _zr = zp - zq;

                const float rij2 = _xr * _xr + _yr * _yr + _zr * _zr;

                const float invrij = rsqrtf( rij2 );

                const float rij = rij2 * invrij;
                const float argwr = max( ( float )0, 1 - rij * invrc );
                const float wr = powf( argwr, powf( 0.5f, -VISCOSITY_S_LEVEL ) );

                const float xr = _xr * invrij;
                const float yr = _yr * invrij;
                const float zr = _zr * invrij;

                const float rdotv =
                    xr * ( up - uq ) +
                    yr * ( vp - vq ) +
                    zr * ( wp - wq );

                const int spid = s + l;
                const int dpid = pid;

                const int arg1 = mask * dpid + ( 1 - mask ) * spid;
                const int arg2 = mask * spid + ( 1 - mask ) * dpid;
                const float myrandnr = Logistic::mean0var1( seed, arg1, arg2 );

                const float strength = aij * argwr + ( - gamma * wr * rdotv + sigmaf * myrandnr ) * wr;
                //if (valid && spid < np_src)
                {
                    xforce += strength * xr;
                    yforce += strength * yr;
                    zforce += strength * zr;
                }
            }
        }
    }

    if( valid ) {
        assert( !isnan( xforce ) );
        assert( !isnan( yforce ) );
        assert( !isnan( zforce ) );

        axayaz[0 + 3 * pid] = xforce;
        axayaz[1 + 3 * pid] = yforce;
        axayaz[2 + 3 * pid] = zforce;
    }
}

void directforces_dpd_cuda_bipartite_nohost(
    const float * const xyzuvw, float * const axayaz, const int np,
    const float * const xyzuvw_src, const int np_src,
    const float aij, const float gamma, const float sigma, const float invsqrtdt,
    const float seed, const int mask, cudaStream_t stream )
{
    if( np == 0 || np_src == 0 ) {
        printf( "warning: directforces_dpd_cuda_bipartite_nohost called with ZERO!\n" );
        return;
    }

    _bipartite_dpd_directforces <<< ( np + 127 ) / 128, 128, 0, stream >>> ( axayaz, np, np_src, seed, mask,
            xyzuvw, xyzuvw_src, 1, aij, gamma, sigma * invsqrtdt );

    CUDA_CHECK( cudaPeekAtLastError() );
}

__global__ __launch_bounds__( 32 * CPB, 16 )
void _dpd_bipforces( const float2 * const xyzuvw, const int np, cudaTextureObject_t texDstStart,
                     cudaTextureObject_t texSrcStart,  cudaTextureObject_t texSrcParticles, const int np_src, const int3 halo_ncells,
                     const float aij, const float gamma, const float sigmaf,
                     const float seed, const uint mask, float * const axayaz )
{
    assert( warpSize == COLS * ROWS );
    assert( blockDim.x == warpSize && blockDim.y == CPB && blockDim.z == 1 );
    assert( ROWS * 3 <= warpSize );

    const uint tid = threadIdx.x;
    const uint subtid = tid % COLS;
    const uint slot = tid / COLS;
    const uint wid = threadIdx.y;

    __shared__ uint volatile starts[CPB][32], scan[CPB][32];

    const int mycid = xmad( blockIdx.x, float(CPB), threadIdx.y );

    if( mycid >= halo_ncells.x * halo_ncells.y * halo_ncells.z ) return;

    uint mycount = 0, myscan = 0;
    if( tid < 27 ) {
        const int dx = ( tid ) % 3;
        const int dy = ( ( tid / 3 ) ) % 3;
        const int dz = ( ( tid / 9 ) ) % 3;

        int xcid = ( mycid % halo_ncells.x ) + dx - 1;
        int ycid = ( ( mycid / halo_ncells.x ) % halo_ncells.y ) + dy - 1;
        int zcid = ( ( mycid / halo_ncells.x / halo_ncells.y ) % halo_ncells.z ) + dz - 1;

        const bool valid_cid =
                ( xcid >= 0 ) && ( xcid < halo_ncells.x ) &&
                ( ycid >= 0 ) && ( ycid < halo_ncells.y ) &&
                ( zcid >= 0 ) && ( zcid < halo_ncells.z );

        xcid = xmin( xsub( halo_ncells.x, 1 ), max( 0, xcid ) );
        ycid = xmin( xsub( halo_ncells.y, 1 ), max( 0, ycid ) );
        zcid = xmin( xsub( halo_ncells.z, 1 ), max( 0, zcid ) );

        const int cid = max( 0, xcid + halo_ncells.x * ( ycid + halo_ncells.y * zcid ) );

        starts[wid][tid] = tex1Dfetch<uint>( texSrcStart, cid );

        myscan = mycount = valid_cid ? ( tex1Dfetch<uint>( texSrcStart, cid + 1 ) - tex1Dfetch<uint>( texSrcStart, cid ) ) : 0u;
    }

    for( int L = 1; L < 32; L <<= 1 )
        myscan += ( tid >= L ) * __shfl_up( myscan, L ) ;

    if( tid < 28 )
        scan[wid][tid] = myscan - mycount;

    const uint dststart = tex1Dfetch<uint>( texDstStart, mycid );
    const uint nsrc = scan[wid][27];
    const uint ndst = xsub( tex1Dfetch<uint>( texDstStart, xadd( mycid, 1 ) ), tex1Dfetch<uint>( texDstStart, mycid ) );

    for( uint d = 0u; d < ndst; d += ROWS ) {
        const uint np1 = xmin( ndst - d, ROWS );
        const uint dpid = xadd( xadd( dststart, d ), slot );
        const int entry = xscale( dpid, 3.f );
        float2 dtmp0 = xyzuvw[entry];
        float2 dtmp1 = xyzuvw[xadd(entry, 1)];
        float2 dtmp2 = xyzuvw[xadd(entry, 2)];

        float3 f = make_float3(0,0,0);

        for( uint s = 0; s < nsrc; s += COLS ) {
            const uint np2 = xmin( xsub(nsrc, s), COLS );
            const uint pid = xadd( s, subtid );
            const uint key9 = xadd( xsel_ge( pid, scan[wid][9            ], 9u, 0u ), xsel_ge( pid, scan[wid][18           ], 9u, 0u ) );
            const uint key3 = xadd( xsel_ge( pid, scan[wid][xadd(key9,3u)], 3u, 0u ), xsel_ge( pid, scan[wid][xadd(key9,6u)], 3u, 0u ) );
            const uint key  = xadd( key9, key3 );
            const uint spid = xsub( xadd( starts[wid][key], pid ), scan[wid][key] );
            const int sentry = xscale( spid, 3.f );

            const float2 stmp0 = tex1Dfetch<float2>( texSrcParticles, sentry );
            const float2 stmp1 = tex1Dfetch<float2>( texSrcParticles, xadd( sentry, 1 ) );
            const float2 stmp2 = tex1Dfetch<float2>( texSrcParticles, xadd( sentry, 2 ) );

            {
                const float _xr = dtmp0.x - stmp0.x;
                const float _yr = dtmp0.y - stmp0.y;
                const float _zr = dtmp1.x - stmp1.x;

                const float rij2 = _xr * _xr + _yr * _yr + _zr * _zr;
                const float invrij = rsqrtf( rij2 );
                const float rij = rij2 * invrij;
                const float argwr = max( ( float )0, 1 - rij );
                const float wr = powf( argwr, powf( 0.5f, -VISCOSITY_S_LEVEL ) );

                const float xr = _xr * invrij;
                const float yr = _yr * invrij;
                const float zr = _zr * invrij;

                const float rdotv =
                    xr * ( dtmp1.y - stmp1.y ) +
                    yr * ( dtmp2.x - stmp2.x ) +
                    zr * ( dtmp2.y - stmp2.y );

                const uint arg1 = xsel_gt( mask, 0u, dpid, spid );
                const uint arg2 = xsel_gt( mask, 0u, spid, dpid );
                const float myrandnr = Logistic::mean0var1( seed, arg1, arg2 );

                const float strength = aij * argwr + ( - gamma * wr * rdotv + sigmaf * myrandnr ) * wr;
                const bool valid = xfcmp_lt( slot, np1 ) * xfcmp_lt( subtid, np2 );

                assert( ( dpid >= 0 && dpid < np && spid >= 0 && spid < np_src ) || ! valid );

                if( valid ) {
                    f.x += strength * xr;
                    f.y += strength * yr;
                    f.z += strength * zr;
                }
            }
        }

        for( uint L = COLS / 2u; L > 0u; L >>= 1 ) {
			f.x += __shfl_xor( f.x, L );
			f.y += __shfl_xor( f.y, L );
			f.z += __shfl_xor( f.z, L );
        }

        const uint c = ( subtid % 3u );
        const float fcontrib = xsel_eq( c, 0u, f.x, xsel_eq( c, 1u, f.y, f.z ) );
        const uint dstpid = xadd( xadd( dststart, d ), slot );

        if( slot < np1 )
            axayaz[ xmad( dstpid, 3.f, c ) ] = fcontrib;
    }
}

void forces_dpd_cuda_bipartite_nohost( cudaStream_t stream, const float2 * const xyzuvw, const int np, cudaTextureObject_t texDstStart,
                                       cudaTextureObject_t texSrcStart, cudaTextureObject_t texSrcParticles, const int np_src,
                                       const int3 halo_ncells,
                                       const float aij, const float gamma, const float sigmaf,
                                       const float seed, const int mask, float * const axayaz )
{
    const int ncells = halo_ncells.x * halo_ncells.y * halo_ncells.z;

    static bool fbip_init = false;

    if( !fbip_init ) {
        CUDA_CHECK( cudaFuncSetCacheConfig( _dpd_bipforces, cudaFuncCachePreferL1 ) );

        fbip_init = true;
    }

    _dpd_bipforces <<< ( ncells + CPB - 1 ) / CPB, dim3( 32, CPB ), 0, stream >>> (
        xyzuvw, np, texDstStart, texSrcStart, texSrcParticles, np_src,
        halo_ncells, aij, gamma, sigmaf, seed, mask,
        axayaz );
}
