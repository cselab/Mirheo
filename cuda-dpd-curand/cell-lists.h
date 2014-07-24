#pragma once

#if 1
// The following encoding/decoding was taken from
// http://fgiesen.wordpress.com/2009/12/13/decoding-morton-codes/
// "Insert" two 0 bits after each of the 10 low bits of x
__device__ inline uint Part1By2(uint x)
{
    x &= 0x000003ff;                  // x = ---- ---- ---- ---- ---- --98 7654 3210
    x = (x ^ (x << 16)) & 0xff0000ff; // x = ---- --98 ---- ---- ---- ---- 7654 3210
    x = (x ^ (x <<  8)) & 0x0300f00f; // x = ---- --98 ---- ---- 7654 ---- ---- 3210
    x = (x ^ (x <<  4)) & 0x030c30c3; // x = ---- --98 ---- 76-- --54 ---- 32-- --10
    x = (x ^ (x <<  2)) & 0x09249249; // x = ---- 9--8 --7- -6-- 5--4 --3- -2-- 1--0
    return x;
} 

// Inverse of Part1By2 - "delete" all bits not at positions divisible by 3
__device__ uint inline Compact1By2(uint x)
{
    x &= 0x09249249;                  // x = ---- 9--8 --7- -6-- 5--4 --3- -2-- 1--0
    x = (x ^ (x >>  2)) & 0x030c30c3; // x = ---- --98 ---- 76-- --54 ---- 32-- --10
    x = (x ^ (x >>  4)) & 0x0300f00f; // x = ---- --98 ---- ---- 7654 ---- ---- 3210
    x = (x ^ (x >>  8)) & 0xff0000ff; // x = ---- --98 ---- ---- ---- ---- 7654 3210
    x = (x ^ (x >> 16)) & 0x000003ff; // x = ---- ---- ---- ---- ---- --98 7654 3210
    return x;
}

__device__ int inline encode(int x, int y, int z) 
{
    return (Part1By2(z) << 2) + (Part1By2(y) << 1) + Part1By2(x);
}

__device__ int3 inline decode(int code)
{
    return make_int3(
	Compact1By2(code >> 0),
	Compact1By2(code >> 1),
	Compact1By2(code >> 2)
	);
}
#else
__device__ int encode(int ix, int iy, int iz) 
{
    const int retval = ix + info.ncells.x * (iy + iz * info.ncells.y);

    assert(retval < info.ncells.x * info.ncells.y * info.ncells.z && retval>=0);

    return retval; 
}
	
__device__ int3 decode(int code)
{
    const int ix = code % info.ncells.x;
    const int iy = (code / info.ncells.x) % info.ncells.y;
    const int iz = (code / info.ncells.x/info.ncells.y);

    return make_int3(ix, iy, iz);
}
#endif

void build_clists(float * const device_xyzuvw, int np, const float rc, 
		      const int xcells, const int ycells, const int zcells,
		      const float xdomainstart, const float ydomainstart, const float zdomainstart,
		      int * const host_order, int * device_startcell, int * device_endcell);
		  
