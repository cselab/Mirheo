#ifndef _TINY_FLOAT_
#define _TINY_FLOAT_

/******************************************************************************
                                 Reintepretation
******************************************************************************/

__forceinline__ __device__ int f2i( float f ) {
        return __float_as_int(f);
}

__forceinline__ __device__ uint f2u( float f )
{
    uint u;
    asm volatile( "mov.b32 %0, %1;" : "=r"( u ) : "f"( f ) );
    return u;
}

__forceinline__ __device__ float i2f( int i ) {
	return __int_as_float(i);
}

__forceinline__ __device__ float u2f( uint u )
{
    float f;
    asm volatile( "mov.b32 %0, %1;" : "=f"( f ) : "r"( u ) );
    return f;
}

/******************************************************************************
                                 Arithmetic
******************************************************************************/

// safety fallback to prevent unknowning usage
template<typename T, typename S> class Do_Not_Use_Tiny_Float_For_Normal_Operations add( T const &t, S const &s );
template<typename T, typename S> class Do_Not_Use_Tiny_Float_For_Normal_Operations scale( T const &t, S const &s );

__forceinline__ __device__ uint add( uint u, uint v ) {
	float a = u2f(u), b = u2f(v), c;
	asm( "add.f32 %0, %1, %2;" : "=f"(c) : "f"(a), "f"(b) );
	return f2u(c);
}

__forceinline__ __device__ int add( int u, int v ) {
	float a = i2f(u), b = i2f(v), c;
	asm( "add.f32 %0, %1, %2;" : "=f"(c) : "f"(a), "f"(b) );
	return f2i(c);
}

__forceinline__ __device__ uint scale( uint u, float s ) {
	float a = u2f(u), b;
	asm( "mul.f32 %0, %1, %2;" : "=f"(b) : "f"(a), "f"(s) );
	return f2u(b);
}

__forceinline__ __device__ int scale( int u, float s ) {
	float a = i2f(u), b;
	asm( "mul.f32 %0, %1, %2;" : "=f"(b) : "f"(a), "f"(s) );
	return f2i(b);
}

#endif
