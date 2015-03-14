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
template<typename T, typename S, typename R> class Do_Not_Use_Tiny_Float_For_Normal_Operations xfma( T const &t, S const &s, R const &r );
template<typename T, typename S> class Do_Not_Use_Tiny_Float_For_Normal_Operations xadd( T const &t, S const &s );
template<typename T, typename S> class Do_Not_Use_Tiny_Float_For_Normal_Operations xsub( T const &t, S const &s );
template<typename T, typename S> class Do_Not_Use_Tiny_Float_For_Normal_Operations xscale( T const &t, S const &s );
template<typename T, typename S> class Do_Not_Use_Tiny_Float_For_Normal_Operations xmin( T const &t, S const &s );
template<typename T, typename S> class Do_Not_Use_Tiny_Float_For_Normal_Operations xmax( T const &t, S const &s );

// u * v + w
__forceinline__ __device__ uint xmad( uint u, float v, uint w ) {
	float a = u2f(u), c = u2f(w), d;
	asm( "fma.rz.f32 %0, %1, %2, %3;" : "=f"(d) : "f"(a), "f"(v), "f"(c) );
	return f2u(d);
}

// u * v + w
__forceinline__ __device__ int xmad( int u, float v, int w ) {
	float a = i2f(u), c = i2f(w), d;
	asm( "fma.rz.f32 %0, %1, %2, %3;" : "=f"(d) : "f"(a), "f"(v), "f"(c) );
	return f2i(d);
}

__forceinline__ __device__ uint xadd( uint u, uint v ) {
	float a = u2f(u), b = u2f(v), c;
	asm( "add.f32 %0, %1, %2;" : "=f"(c) : "f"(a), "f"(b) );
	return f2u(c);
}

__forceinline__ __device__ int xadd( int u, int v ) {
	float a = i2f(u), b = i2f(v), c;
	asm( "add.f32 %0, %1, %2;" : "=f"(c) : "f"(a), "f"(b) );
	return f2i(c);
}

__forceinline__ __device__ uint xsub( uint u, uint v ) {
	float a = u2f(u), b = u2f(v), c;
	asm( "sub.f32 %0, %1, %2;" : "=f"(c) : "f"(a), "f"(b) );
	return f2u(c);
}

__forceinline__ __device__ int xsub( int u, int v ) {
	float a = i2f(u), b = i2f(v), c;
	asm( "sub.f32 %0, %1, %2;" : "=f"(c) : "f"(a), "f"(b) );
	return f2i(c);
}

__forceinline__ __device__ uint xscale( uint u, float s ) {
	float a = u2f(u), b;
	asm( "mul.f32 %0, %1, %2;" : "=f"(b) : "f"(a), "f"(s) );
	return f2u(b);
}

__forceinline__ __device__ int xscale( int u, float s ) {
	float a = i2f(u), b;
	asm( "mul.f32 %0, %1, %2;" : "=f"(b) : "f"(a), "f"(s) );
	return f2i(b);
}

__forceinline__ __device__ int xmin( int u, int v ) {
	float a = i2f(u), b = i2f(v), c;
	asm( "min.f32 %0, %1, %2;" : "=f"(c) : "f"(a), "f"(b) );
	return f2i(c);
}

__forceinline__ __device__ uint xmin( uint u, uint v ) {
	float a = u2f(u), b = u2f(v), c;
	asm( "min.f32 %0, %1, %2;" : "=f"(c) : "f"(a), "f"(b) );
	return f2u(c);
}

__forceinline__ __device__ int xmax( int u, int v ) {
	float a = i2f(u), b = i2f(v), c;
	asm( "max.f32 %0, %1, %2;" : "=f"(c) : "f"(a), "f"(b) );
	return f2i(c);
}

__forceinline__ __device__ uint xmax( uint u, uint v ) {
	float a = u2f(u), b = u2f(v), c;
	asm( "max.f32 %0, %1, %2;" : "=f"(c) : "f"(a), "f"(b) );
	return f2u(c);
}

#endif
