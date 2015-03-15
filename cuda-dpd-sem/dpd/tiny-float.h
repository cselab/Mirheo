#ifndef _TINY_FLOAT_
#define _TINY_FLOAT_

/******************************************************************************
                                 Reintepretation
******************************************************************************/

// Prevent unknowning usage due to type promotion
template<typename T> __device__ class Automatic_Type_Promotion_Not_Allowed f2i( T const &t );
template<typename T> __device__ class Automatic_Type_Promotion_Not_Allowed f2u( T const &t );
template<typename T> __device__ class Automatic_Type_Promotion_Not_Allowed i2f( T const &t );
template<typename T> __device__ class Automatic_Type_Promotion_Not_Allowed u2f( T const &t );

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

// Prevent unknowning usage due to type promotion
template<typename T, typename S, typename R> __device__ class Do_Not_Use_Tiny_Float_For_Normal_Operations xmad( T const &t, S const &s, R const &r );
template<typename T, typename S> __device__ class Do_Not_Use_Tiny_Float_For_Normal_Operations xadd( T const &t, S const &s );
template<typename T, typename S> __device__ class Do_Not_Use_Tiny_Float_For_Normal_Operations xsub( T const &t, S const &s );
template<typename T, typename S> __device__ class Do_Not_Use_Tiny_Float_For_Normal_Operations xscale( T const &t, S const &s );
template<typename T, typename S> __device__ class Do_Not_Use_Tiny_Float_For_Normal_Operations xmin( T const &t, S const &s );
template<typename T, typename S> __device__ class Do_Not_Use_Tiny_Float_For_Normal_Operations xmax( T const &t, S const &s );

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

__forceinline__ __device__ uint xadd( float u, float v ) {
	float c;
	asm( "add.f32 %0, %1, %2;" : "=f"(c) : "f"(u), "f"(v) );
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

///******************************************************************************
//                                 Compare
//******************************************************************************/
//
//// result: (i>j)
//__forceinline__ __device__ bool xcmpgt( uint i, uint j ) {
//	uint r;
//	float a = u2f(i), b = u2f(j);
//	asm( "{ \
//			.reg .pred p;\
//			setp.gt.f32 p, %1, %2;\
//			selp.u32    %0, 1, 0, p;\
//          }" : "=r"(r) : "f"(a), "f"(b) );
//	return bool(r);
//}
//
//__forceinline__ __device__ uint xselgt( uint i, uint j, uint u, uint v ) {
//	float r;
//	float a = u2f(i), b = u2f(j), c = u2f(u), d = u2f(v);
//	asm( "{ \
//			.reg .pred p;\
//			setp.gt.f32 p, %1, %2;\
//			selp.f32    %0, %3, %4, p;\
//          }" : "=f"(r) : "f"(a), "f"(b), "f"(c), "f"(d) );
//	return f2u(r);
//}

/******************************************************************************
                                 Branch
******************************************************************************/

#define __CONCAT__(x,y) x##y
#define CONCAT(x,y) __CONCAT__(x,y)
#define UNIQUE(variable) CONCAT(variable##_,__LINE__)

#define xfor(i,i_beg,i_end,inc,LOOP) \
uint i, init_##LOOP = i_beg; \
const float f_##LOOP = u2f(i_beg); \
const float g_##LOOP = u2f(i_end); \
asm volatile( "{" ); \
asm volatile( ".reg .pred p,q;" ); \
asm volatile( "setp.geu.f32 p, %0, %1;" : : "f"(f_##LOOP), "f"(g_##LOOP) : "memory" ); \
asm volatile( "mov.b32 %0, %1;" : "=r"(i) : "r"(init_##LOOP) : "memory" ); \
asm volatile( "@p bra " #LOOP "_END;" : : : "memory"  ); \
asm volatile( #LOOP "_BEG:" : : : "memory");

#define xendfor(i,i_beg,i_end,inc,LOOP) \
const float i##_##LOOP = u2f(i) + u2f(inc); \
i = f2u( i##_##LOOP ); \
asm volatile( "setp.lt.f32 q, %0, %1;" : : "f"(i##_##LOOP), "f"(g_##LOOP) : "memory" ); \
asm volatile( "@q bra " #LOOP "_BEG;" : : : "memory" ); \
asm volatile( #LOOP "_END:" : : : "memory" );\
asm volatile( "}" );

//__global__ void foo( const uint n ) {
//	volatile uint sum = 0;
//	xfor(i,0,n,1,LOOP)
//	sum += i * i;
//	xendfor(i,0,n,1,LOOP)
//	printf("%u\n",sum);
//}

#endif
