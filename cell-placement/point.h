/*
 * Vector.h
 *
 *  Created on: Nov 12, 2014
 *      Author: ytang
 */

#ifndef POINT_H_
#define POINT_H_

#include<array>
#include<cmath>
#include<algorithm>
#include<cassert>
#include<iostream>

namespace ermine {

using uint = unsigned int;

template<typename T1, typename T2> struct same_type { static bool const yes = false; };
template<typename T> struct same_type<T,T> { static bool const yes = true; };

// Reference:
// Expression template: http://en.wikipedia.org/wiki/Expression_templates
// CRTP: http://en.wikipedia.org/wiki/Curiously_recurring_template_pattern

/*---------------------------------------------------------------------------
                              Interface
---------------------------------------------------------------------------*/

template<class VEX, typename SCALAR, uint D>
struct VecExp {
	using TYPE_ = SCALAR;
	static const uint D_ = D;

	// the only work in constructor is type checking
	inline VecExp() {
		static_assert( same_type<TYPE_, typename VEX::TYPE_>::yes, "Vector element type mismatch" );
		static_assert( D_ == VEX::D_, "Vector dimensionality mismatch" );
	}

	// dereferencing using static polymorphism
	inline SCALAR operator[] (uint i) const {
		assert( i < D );
		return static_cast<VEX const &>(*this)[i];
	}

	inline operator VEX      & ()       { return static_cast<VEX      &>(*this); }
	inline operator VEX const& () const { return static_cast<VEX const&>(*this); }

	inline uint d() const { return D_; }
};

/*---------------------------------------------------------------------------
                                 Container
---------------------------------------------------------------------------*/

template<typename SCALAR, uint D=3U>
struct Vector : public VecExp<Vector<SCALAR,D>, SCALAR, D> {
protected:
	SCALAR x[D];
public:
	using TYPE_ = SCALAR;
	static const uint D_ = D;

	// default constructor
	inline Vector() {}
	// construct from scalar constant
	inline Vector(SCALAR const s) { for(uint i = 0 ; i < D ; i++) x[i] = s; }
	inline Vector(int    const s) { for(uint i = 0 ; i < D ; i++) x[i] = SCALAR(s); }
	inline Vector(uint   const s) { for(uint i = 0 ; i < D ; i++) x[i] = SCALAR(s); }
	inline Vector(long   const s) { for(uint i = 0 ; i < D ; i++) x[i] = SCALAR(s); }
	inline Vector(ulong  const s) { for(uint i = 0 ; i < D ; i++) x[i] = SCALAR(s); }
	// construct from C-array
	inline Vector(SCALAR const *ps) { for(uint i = 0 ; i < D ; i++) x[i] = *ps++; }
	// construct from parameter pack
	// 'head' differentiate it from constructing from vector expression
	template<typename ...T> inline Vector(SCALAR const head, T const ... tail ) {
		std::array<TYPE_,D_> s( { head, tail... } );
		for(uint i = 0 ; i < D ; i++) x[i] = s[i];
	}
	// construct from any vector expression
	template<class E> inline Vector( const VecExp<E,SCALAR,D> &u ) {
		for(uint i = 0 ; i < D ; i++) x[i] = u[i];
	}

	// Vector must be assignable, while other expressions may not
	inline SCALAR      & operator [] (uint i)       { assert( i < D ); return x[i]; }
	inline SCALAR const& operator [] (uint i) const { assert( i < D ); return x[i]; }

	// STL-style direct data accessor
	inline SCALAR      * data()       { return x; }
	inline SCALAR const* data() const { return x; }

	// assign from any vector expression
	template<class E> inline Vector & operator += ( const VecExp<E,SCALAR,D> &u ) {
		for(uint i = 0 ; i < D ; i++) x[i] += u[i];
		return *this;
	}
	template<class E> inline Vector & operator -= ( const VecExp<E,SCALAR,D> &u ) {
		for(uint i = 0 ; i < D ; i++) x[i] -= u[i];
		return *this;
	}
	template<class E> inline Vector & operator *= ( const VecExp<E,SCALAR,D> &u ) {
		for(uint i = 0 ; i < D ; i++) x[i] *= u[i];
		return *this;
	}
	template<class E> inline Vector & operator /= ( const VecExp<E,SCALAR,D> &u ) {
		for(uint i = 0 ; i < D ; i++) x[i] /= u[i];
		return *this;
	}
	// conventional vector-scalar operators
	inline Vector & operator += ( SCALAR const u ) {
		for(uint i = 0 ; i < D ; i++) x[i] += u;
		return *this;
	}
	inline Vector & operator -= ( SCALAR const u ) {
		for(uint i = 0 ; i < D ; i++) x[i] -= u;
		return *this;
	}
	inline Vector & operator *= ( SCALAR const u ) {
		for(uint i = 0 ; i < D ; i++) x[i] *= u;
		return *this;
	}
	inline Vector & operator /= ( SCALAR const u ) {
		return operator *= ( SCALAR(1)/u );
	}

	// special vectors
	static inline Vector const & zero() {
		Vector zero_(0);
		return zero_;
	}
};

/*---------------------------------------------------------------------------
                         Arithmetic Functors
---------------------------------------------------------------------------*/

template<class E1, class E2, typename SCALAR, uint D>
struct VecAdd: public VecExp<VecAdd<E1,E2,SCALAR,D>, SCALAR, D> {
	inline VecAdd( VecExp<E1,SCALAR,D> const& u, VecExp<E2,SCALAR,D> const& v ) : u_(u), v_(v) {}
	inline SCALAR operator [] (uint i) const { return u_[i] + v_[i]; }
protected:
	E1 const& u_;
	E2 const& v_;
};

template<class E1, class E2, typename SCALAR, uint D>
struct VecSub: public VecExp<VecSub<E1,E2,SCALAR,D>, SCALAR, D> {
	inline VecSub( VecExp<E1,SCALAR,D> const& u, VecExp<E2,SCALAR,D> const& v ) : u_(u), v_(v) {}
	inline SCALAR operator [] (uint i) const { return u_[i] - v_[i]; }
protected:
	E1 const& u_;
	E2 const& v_;
};

template<class E, typename SCALAR, uint D>
struct VecNeg: public VecExp<VecNeg<E,SCALAR,D>, SCALAR, D> {
	inline VecNeg( VecExp<E,SCALAR,D> const& u ) : u_(u) {}
	inline SCALAR operator [] (uint i) const { return -u_[i]; }
protected:
	E const& u_;
};

template<class E1, class E2, typename SCALAR, uint D>
struct VecMul: public VecExp<VecMul<E1,E2,SCALAR,D>, SCALAR, D> {
	inline VecMul( VecExp<E1,SCALAR,D> const& u, VecExp<E2,SCALAR,D> const& v ) : u_(u), v_(v) {}
	inline SCALAR operator [] (uint i) const { return u_[i] * v_[i]; }
protected:
	E1 const& u_;
	E2 const& v_;
};

template<class E, typename SCALAR, uint D>
struct VecScale: public VecExp<VecScale<E,SCALAR,D>, SCALAR, D> {
	inline VecScale( VecExp<E,SCALAR,D> const& u, SCALAR const a ) : u_(u), a_(a) {}
	inline SCALAR operator [] (uint i) const { return u_[i] * a_; }
protected:
	E      const& u_;
	SCALAR const  a_;
};

template<class E, typename SCALAR, uint D>
struct VecAddScalar: public VecExp<VecAddScalar<E,SCALAR,D>, SCALAR, D> {
	inline VecAddScalar( VecExp<E,SCALAR,D> const& u, SCALAR const a ) : u_(u), a_(a) {}
	inline SCALAR operator [] (uint i) const { return u_[i] + a_; }
protected:
	E      const& u_;
	SCALAR const  a_;
};

template<class E, typename SCALAR, uint D>
struct VecSubScalar: public VecExp<VecSubScalar<E,SCALAR,D>, SCALAR, D> {
	inline VecSubScalar( VecExp<E,SCALAR,D> const& u, SCALAR const a ) : u_(u), a_(a) {}
	inline SCALAR operator [] (uint i) const { return u_[i] - a_; }
protected:
	E      const& u_;
	SCALAR const  a_;
};

template<class E1, class E2, typename SCALAR, uint D>
struct VecDiv: public VecExp<VecDiv<E1,E2,SCALAR,D>, SCALAR, D> {
	inline VecDiv( VecExp<E1,SCALAR,D> const& u, VecExp<E2,SCALAR,D> const& v ) : u_(u), v_(v) {}
	inline SCALAR operator [] (uint i) const { return u_[i] / v_[i]; }
protected:
	E1 const& u_;
	E2 const& v_;
};

template<class E, typename SCALAR, uint D>
struct vexpr_rcp: public VecExp<vexpr_rcp<E,SCALAR,D>, SCALAR, D> {
	inline vexpr_rcp( VecExp<E,SCALAR,D> const& u ) : u_(u) {}
	inline SCALAR operator [] (uint i) const { return SCALAR(1)/u_[i]; }
protected:
	E const& u_;
};

template<class E, typename SCALAR, uint D>
struct VecScaleRcp: public VecExp<VecScaleRcp<E,SCALAR,D>, SCALAR, D> {
	inline VecScaleRcp( SCALAR const a, VecExp<E,SCALAR,D> const& u ) : a_(a), u_(u) {}
	inline SCALAR operator [] (uint i) const { return a_ / u_[i]; }
protected:
	SCALAR const  a_;
	E      const& u_;
};

template<class E1, class E2, typename SCALAR>
struct VecCross: public VecExp<VecCross<E1,E2,SCALAR>, SCALAR, 3U> {
	inline VecCross( VecExp<E1,SCALAR,3U> const& u, VecExp<E2,SCALAR,3U> const& v ) : u_(u), v_(v) {}
	inline SCALAR operator [] (uint i) const { return u_[(i+1U)%3U] * v_[(i+2U)%3U] - u_[(i+2U)%3U] * v_[(i+1U)%3U]; }
protected:
	E1 const& u_;
	E2 const& v_;
};

template<class E, class OP, typename SCALAR, uint D>
struct VecApply1: public VecExp<VecApply1<E,OP,SCALAR,D>, SCALAR, D> {
	inline VecApply1( VecExp<E,SCALAR,D> const& u, OP const op ) : u_(u), o_(op) {}
	inline SCALAR operator [] (uint i) const { return o_( u_[i] ); }
protected:
	E  const& u_;
	OP const  o_;
};

template<class E1, class E2, class OP, typename SCALAR, uint D>
struct VecApply2: public VecExp<VecApply2<E1,E2,OP,SCALAR,D>, SCALAR, D> {
	inline VecApply2( VecExp<E1,SCALAR,D> const& u, VecExp<E2,SCALAR,D> const& v, OP const op ) : u_(u), v_(v), o_(op) {}
	inline SCALAR operator [] (uint i) const { return o_( u_[i], v_[i] ); }
protected:
	E1 const& u_;
	E2 const& v_;
	OP const  o_;
};

/*---------------------------------------------------------------------------
                         Operator Overloads
---------------------------------------------------------------------------*/

template<class E1, class E2, typename SCALAR, uint D> inline
VecAdd<E1, E2, SCALAR, D> operator + ( VecExp<E1,SCALAR,D> const &u, VecExp<E2,SCALAR,D> const &v ) {
	return VecAdd<E1, E2, SCALAR, D>( u, v );
}

template<class E1, class E2, typename SCALAR, uint D> inline
VecSub<E1, E2, SCALAR, D> operator - ( VecExp<E1,SCALAR,D> const &u, VecExp<E2,SCALAR,D> const &v ) {
	return VecSub<E1, E2, SCALAR, D>( u, v );
}

template<class E, typename SCALAR, uint D> inline
VecNeg<E, SCALAR, D> operator - ( VecExp<E,SCALAR,D> const &u ) {
	return VecNeg<E, SCALAR, D>( u );
}

template<class E1, class E2, typename SCALAR, uint D> inline
VecMul<E1, E2, SCALAR, D> operator * ( VecExp<E1,SCALAR,D> const &u, VecExp<E2,SCALAR,D> const &v ) {
	return VecMul<E1, E2, SCALAR, D>( u, v );
}

template<class E, typename SCALAR, uint D> inline
VecScale<E, SCALAR, D> operator * ( VecExp<E,SCALAR,D> const &u, SCALAR const a ) {
	return VecScale<E, SCALAR, D>( u, a );
}

template<class E, typename SCALAR, uint D> inline
VecScale<E, SCALAR, D> operator * ( SCALAR const a, VecExp<E,SCALAR,D> const &u ) {
	return VecScale<E, SCALAR, D>( u, a );
}

template<class E, typename SCALAR, uint D> inline
VecScale<E, SCALAR, D> operator / ( VecExp<E,SCALAR,D> const &u, SCALAR const a ) {
	return VecScale<E, SCALAR, D>( u, SCALAR(1)/a );
}

template<class E, typename SCALAR, uint D> inline
VecAddScalar<E, SCALAR, D> operator + ( VecExp<E,SCALAR,D> const &u, SCALAR const a ) {
	return VecAddScalar<E, SCALAR, D>( u, a );
}

template<class E, typename SCALAR, uint D> inline
VecAddScalar<E, SCALAR, D> operator + ( SCALAR const a, VecExp<E,SCALAR,D> const &u ) {
	return VecAddScalar<E, SCALAR, D>( u, a );
}

template<class E, typename SCALAR, uint D> inline
VecSubScalar<E, SCALAR, D> operator - ( VecExp<E,SCALAR,D> const &u, SCALAR const a ) {
	return VecSubScalar<E, SCALAR, D>( u, a );
}

template<class E, typename SCALAR, uint D> inline
VecSubScalar<E, SCALAR, D> operator - ( SCALAR const a, VecExp<E,SCALAR,D> const &u ) {
	return VecAddScalar<E, SCALAR, D>( -u, a );
}

template<class E1, class E2, typename SCALAR, uint D> inline
VecDiv<E1, E2, SCALAR, D> operator / ( VecExp<E1,SCALAR,D> const &u, VecExp<E2,SCALAR,D> const &v ) {
	return VecDiv<E1, E2, SCALAR, D>( u, v );
}

template<class E, typename SCALAR, uint D> inline
VecScaleRcp<E, SCALAR, D> operator / ( SCALAR const a, VecExp<E,SCALAR,D> const &u ) {
	return VecScaleRcp<E, SCALAR, D>( a, u );
}

/*---------------------------------------------------------------------------
                         Math functions
---------------------------------------------------------------------------*/

template<class E1, class E2, typename SCALAR> inline
VecCross<E1, E2, SCALAR> cross( VecExp<E1,SCALAR,3U> const &u, VecExp<E2,SCALAR,3U> const &v ) {
	return VecCross<E1, E2, SCALAR>( u, v );
}

// generic reduction template
template<class E, class OP, typename SCALAR, uint D> inline
SCALAR reduce( VecExp<E,SCALAR,D> const &u, OP const & op ) {
	SCALAR core( u[0] );
	for(uint i = 1 ; i < D ; i++) core = op( core, u[i] );
	return core;
}

// biggest element within a vector
template<class E, typename SCALAR, uint D> inline
SCALAR max( VecExp<E,SCALAR,D> const &u ) {
	return reduce( u, [](SCALAR a, SCALAR b){return a>b?a:b;} );
}

// smallest element within a vector
template<class E, typename SCALAR, uint D> inline
SCALAR min( VecExp<E,SCALAR,D> const &u ) {
	return reduce( u, [](SCALAR a, SCALAR b){return a<b?a:b;} );
}

// smallest element within a vector
template<class E, typename SCALAR, uint D> inline
SCALAR sum( VecExp<E,SCALAR,D> const &u ) {
	return reduce( u, [](SCALAR a, SCALAR b){return a+b;} );
}

// smallest element within a vector
template<class E, typename SCALAR, uint D> inline
SCALAR mean( VecExp<E,SCALAR,D> const &u ) {
	return sum(u) / SCALAR(D);
}

// inner product
template<class E1, class E2, typename SCALAR, uint D> inline
SCALAR dot( VecExp<E1,SCALAR,D> const &u, VecExp<E2,SCALAR,D> const &v ) {
	return sum( u * v );
}

// square of L2 norm
template<class E, typename SCALAR, uint D> inline
SCALAR normsq( VecExp<E,SCALAR,D> const &u ) {
	return sum( u * u );
}

// L2 norm
template<class E, typename SCALAR, uint D> inline
SCALAR norm( VecExp<E,SCALAR,D> const &u ) {
	return std::sqrt( normsq(u) );
}

template<class E, typename SCALAR, uint D> inline
VecScale<E, SCALAR, D> normalize( VecExp<E,SCALAR,D> const &u ) {
	return VecScale<E, SCALAR, D>( u, SCALAR(1)/norm(u) );
}

// element-wise arbitrary function applied for each element
template<class E, class OP, typename SCALAR, uint D> inline
VecApply1<E, OP, SCALAR, D> apply( VecExp<E,SCALAR,D> const &u, OP const& op ) {
	return VecApply1<E, OP, SCALAR, D>( u, op );
}

// element-wise arbitrary function applied element-wisely between 2 vectors
template<class E1, class E2, class OP, typename SCALAR, uint D> inline
VecApply2<E1, E2, OP, SCALAR, D> apply( VecExp<E1,SCALAR,D> const &u, VecExp<E2,SCALAR,D> const &v, OP const& op ) {
	return VecApply2<E1, E2, OP, SCALAR, D>( u, v, op );
}

// element-wise flooring down
template<class E, typename SCALAR, uint D> inline
VecApply1<E, SCALAR(*)(SCALAR), SCALAR, D> floor( VecExp<E,SCALAR,D> const &u ) {
	return VecApply1<E, SCALAR(*)(SCALAR), SCALAR, D>( u, [](SCALAR s){return std::floor(s);} );
}

// element-wise ceiling up
template<class E, typename SCALAR, uint D> inline
VecApply1<E, SCALAR(*)(SCALAR), SCALAR, D> ceil( VecExp<E,SCALAR,D> const &u ) {
	return VecApply1<E, SCALAR(*)(SCALAR), SCALAR, D>( u, [](SCALAR s){return std::ceil(s);} );
}

// element-wise pick bigger
template<class E1, class E2, typename SCALAR, uint D> inline
VecApply2<E1, E2, SCALAR(*)(SCALAR,SCALAR), SCALAR, D> max( VecExp<E1,SCALAR,D> const &u, VecExp<E2,SCALAR,D> const &v ) {
	return VecApply2<E1, E2, SCALAR(*)(SCALAR,SCALAR), SCALAR, D>( u, v, [](SCALAR s,SCALAR t){ return s>t?s:t;} );
}

// element-wise pick smaller
template<class E1, class E2, typename SCALAR, uint D> inline
VecApply2<E1, E2, SCALAR(*)(SCALAR,SCALAR), SCALAR, D> min( VecExp<E1,SCALAR,D> const &u, VecExp<E2,SCALAR,D> const &v ) {
	return VecApply2<E1, E2, SCALAR(*)(SCALAR,SCALAR), SCALAR, D>( u, v, [](SCALAR s,SCALAR t){ return s<t?s:t;} );
}

template<class E1, class E2, class E3, typename SCALAR, uint D> inline
auto clamp( VecExp<E1,SCALAR,D> const &u, VecExp<E2,SCALAR,D> const &l, VecExp<E3,SCALAR,D> const &r )
-> decltype( min(max(u,l),r) )
{
	return min(max(u,l),r);
}

/*---------------------------------------------------------------------------
                         I/O functions
---------------------------------------------------------------------------*/

template<class E, typename SCALAR, uint D> inline
std::ostream& operator <<( std::ostream &out, VecExp<E,SCALAR,D> const &u ) {
	for(int i = 0 ; i < D ; i++) out<<u[i]<<' ';
	return out;
}

template<typename SCALAR, uint D> inline
std::istream& operator >>( std::istream &in, Vector<SCALAR,D> &u ) {
	for(int i = 0 ; i < D ; i++) in>>u[i];
	return in;
}

/*---------------------------------------------------------------------------
                             Matrix
---------------------------------------------------------------------------*/


template<typename REAL, uint M=3, uint N=M>
struct Matrix
{
	Matrix() {
		for(uint i=0;i<M;i++) for(uint j=0;j<N;j++) _e[i][j] = 0.;
	}
	Matrix( const Matrix& other ) {
		*this = other;
	}
	Matrix( const Vector<REAL,M>& other ) {
		for(uint i=0;i<M;i++) _e[i][0] = other[i];
	}
	Matrix& operator = ( const Matrix& other ) {
		for(uint i=0;i<M;i++) for(uint j=0;j<N;j++) _e[i][j] = other._e[i][j];
		return *this;
	}

	inline const REAL& operator () (uint row, uint col) const { assert(row<M&&col<N); return _e[row][col]; }
	inline REAL& operator () (uint row, uint col) { assert(row<M&&col<N); return _e[row][col]; }
	inline int m() const { return M; }
	inline int n() const { return N; }
protected:
	REAL _e[M][N];
};

template<typename REAL, uint M=3, uint N=M>
struct Identity: public Matrix<REAL,M,N>
{
	Identity() {
		static_assert(M==N,"Identity matrix must be square");
		for(uint i=0;i<M;i++) for(uint j=0;j<N;j++) this->_e[i][j] = (i==j) ? 1.0 : 0.0;
	}
};

template<typename REAL, uint M1, uint N1, uint M2, uint N2>
Matrix<REAL,M1,N2> operator * ( const Matrix<REAL,M1,N1> &m1, const Matrix<REAL,M2,N2> &m2 )
{
	static_assert( N1 == M2, "matrix inner dimension mismatch" );
	Matrix<REAL,M1,N2> r;
	for(uint i=0;i<M1;i++)
		for(uint j=0;j<N2;j++)
			for(uint k=0;k<N1;k++)
				r(i,j) += m1(i,k) * m2(k,j);
	return r;
}

template<typename REAL, uint M, uint N>
Matrix<REAL,M,N> operator + ( const Matrix<REAL,M,N> &m1, const Matrix<REAL,M,N> &m2 )
{
	Matrix<REAL,M,N> r;
	for(uint i=0;i<M;i++)
		for(uint j=0;j<N;j++)
			r(i,j) = m1(i,j) + m2(i,j);
	return r;
}

template<typename REAL, uint M, uint N>
Matrix<REAL,M,N> operator * ( const Matrix<REAL,M,N> &m, REAL s )
{
	Matrix<REAL,M,N> r;
	for(uint i=0;i<M;i++)
		for(uint j=0;j<N;j++)
			r(i,j) = m(i,j) * s;
	return r;
}

template<typename REAL, uint M>
Matrix<REAL,M,M> transpose( const Matrix<REAL,M,M> &m )
{
	Matrix<REAL,M,M> r;
	for(uint i=0;i<M;i++)
		for(uint j=0;j<M;j++)
			r(i,j) = m(j,i);
	return r;
}

template<class E, typename REAL, uint D>
Vector<REAL,D> operator * ( Matrix<REAL,D,D> const& m, VecExp<E,REAL,D> const& v ) {
	Vector<REAL,D> r(0.);
	for(uint i = 0 ; i < D ; i++) for(uint j = 0 ; j < D ; j++) r[i] += m(i,j) * v[j];
	return r;
}


}

#endif /* POINT_H_ */
