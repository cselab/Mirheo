/*
 *  main.cpp
 *  Part of CTC/cell-placement/
 *
 *  Created and authored by Diego Rossinelli on 2014-12-18.
 *  Major revision by Yu-Hang Tang on
 *  Copyright 2015. All rights reserved.
 *
 *  Users are NOT authorized
 *  to employ the present software for their own publications
 *  before getting a written permission from the author of this file.
 */

#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <iostream>
#include <fstream>
#include <vector>
#include <map>
#include <algorithm>
#include <omp.h>

#include "point.h"

using namespace std;
using namespace ermine;

using point = Vector<double,3>;
using matrix = Matrix<double,4,4>;
using identity = Identity<double,4,4>;

void verify( string path2ic )
{
    printf( "VERIFYING <%s>\n", path2ic.c_str() );

    FILE * f = fopen( path2ic.c_str(), "r" );

    bool isgood = true;

    while( isgood ) {
        double tmp[19];
        for( int c = 0; c < 19; ++c ) {
            int retval = fscanf( f, "%lf", tmp + c );

            isgood &= retval == 1;
        }

        if( isgood ) {
            printf( "reading: " );

            for( int c = 0; c < 19; ++c )
                printf( "%lf ", tmp[c] );

            printf( "\n" );
        }
    }

    fclose( f );

    printf( "========================================\n\n\n\n" );
}

// low-dimensional RBC model
struct LD_RBC {
	constexpr static double r = 2.2;
	constexpr static int n = 8;
	std::array<point,n> vert;
	point xmin, xmax;
	matrix transform;
	LD_RBC() {
		constexpr double L = 2.0;
		for(int i=0;i<n;i++) {
			vert[i] = point( L * cos(i*2.f*M_PI/n), L * sin(i*2.f*M_PI/n), 0. );
		}
	}
};

struct CTC {
	constexpr static double r = 10.0;
	constexpr static int n = 1;
	std::array<point,n> vert;
	point xmin, xmax;
	matrix transform;
	CTC() {
		vert[0] = point(0);
	}
};

template<typename P> void set_minmax( P &p ) {
	p.xmin = 1e20;
	p.xmax = 1e-20;
	for(auto &v: p.vert) {
		for(int i=0;i<3;i++) {
			p.xmin[i] = min( p.xmin[i], v[i] );
			p.xmax[i] = max( p.xmax[i], v[i] );
		}
	}
}

template<typename P> void randomize( P &p, point global_ext  ) {
	point com(0);
	for( auto const &v: p.vert ) com += v;
	com /= double(p.n);

	double maxr = 0.;
	for( auto const &v: p.vert ) maxr = max( maxr, norm(v - com));
	maxr += p.r;

	point dx = ( global_ext - 2 * maxr ) * point( drand48(), drand48(), drand48() ) +  maxr;

	point da( 0.25 * ( drand48() - 0.5 ) * 2 * M_PI,
	          M_PI * 0.5 + 0.25 * ( drand48() * 2 - 1 ) * M_PI,
	          0.25 * ( drand48() - 0.5 ) * 2 * M_PI );

	matrix T1 = identity(), T2 = identity(), Rx = identity(), Ry = identity(), Rz = identity();

	// move to COM
	T1(0,3) = -com[0], T1(1,3) = -com[1], T1(2,3) = -com[2];

	// rotate randonmly
	Rx(0,0) = 1.; Rx(1,1) = cos(da[0]); Rx(1,2) =-sin(da[0]); Rx(2,1) = sin(da[0]); Rx(2,2) = cos(da[0]); Rx(3,3) = 1.;

	Ry(1,1) = 1.; Ry(0,0) = cos(da[1]); Ry(0,2) = sin(da[1]); Ry(2,0) =-sin(da[1]); Ry(2,2) = cos(da[1]); Ry(3,3) = 1.;

	Rz(2,2) = 1.; Rz(0,0) = cos(da[2]); Rz(0,1) =-sin(da[2]); Rz(1,0) = sin(da[2]); Rz(1,1) = cos(da[2]); Rz(3,3) = 1.;

	// move to random point in global, margin = maxr
	T2(0,3) = dx[0], T2(1,3) = dx[1], T2(2,3) = dx[2];

	p.transform = T2 * Rz * Ry * Rx * T1;

	for( auto &v: p.vert ) {
		Vector<double,4> u( v[0], v[1], v[2], 1. );
		u = p.transform * u;
		v[0] = u[0], v[1] = u[1], v[2] = u[2]; // div by u[3] not needed because no scaling
	}

	set_minmax( p );
}

struct cell {
	int64_t i, j, k;
	cell(int64_t i_, int64_t j_, int64_t k_) : i(i_), j(j_), k(k_) {}
	friend inline bool operator < ( cell const& x, cell const &y ) {
		return (x.i<y.i) || (x.i==y.i&&x.j<y.j) || (x.i==y.i&&x.j==y.j&&x.k<y.k);
	}
};

struct cell_list : public map<cell,vector<int64_t> > {
	double spacing, margin;

	cell_list( double s, double m ) : spacing(s), margin(m) {}

	inline cell p2c(point p) {
		point ijk = floor( p / spacing );
		return cell( ijk[0], ijk[1], ijk[2] );
	}
	inline void put( point xmin, point xmax, int64_t idx ) {
		cell lower = p2c( xmin - margin );
		cell upper = p2c( xmax + margin );
		for( int64_t i = lower.i ; i <= upper.i ; i++ )
			for( int64_t j = lower.j ; j <= upper.j ; j++ )
				for( int64_t k = lower.k ; k <= upper.k ; k++ )
					(*this)[cell(i,j,k)].push_back(idx);
	}
};

template<typename P, typename Q> bool collides( P const& p, Q const &q, const double tol = 0 ) {
	constexpr double r = p.r + q.r;
	for(int i=0;i<p.n;i++)
		for(int j=0;j<q.n;j++) {
			if ( normsq( p.vert[i] - q.vert[j] ) < (r+tol)*(r+tol) ) return true;
		}
	return false;
}

int main( int argc, const char ** argv )
{
    if( argc < 4 ) {
        printf( "usage: ./cell-placement <xdomain-extent> <ydomain-extent> <zdomain-extent> [vol_fraction] \n" );
        exit( -1 );
    }

    point domainextent( atof(argv[1]), atof(argv[2]), atof(argv[3]) );

    printf( "domain extent: %lf %lf %lf\n", domainextent[0], domainextent[1], domainextent[2] );

    double vol = domainextent[0] * domainextent[1] * domainextent[2];
    int64_t maxn = argc >= 5 ? ( vol * atof( argv[4] ) / 92.45 ) : 0xFFFFFFFFFFLL;

    printf(" max number of RBCs: %ld\n", maxn );
    bool failed = false;

    cell_list      clist_rbc(12, 5);
    cell_list      clist_ctc(48,12);
    vector<LD_RBC> result_rbc;
    vector<CTC>    result_ctc;

    double t1 = omp_get_wtime();

    while( !failed && result_rbc.size() + result_ctc.size() < maxn ) {
        const int maxattempts = 1000000;

        int attempt = 0;
        do {
            const int type = 0;//(int)(drand48() >= 0.25);

            LD_RBC t;
            randomize( t, domainextent );

            bool colliding = false;

            point center = ( t.xmin + t.xmax ) * 0.5;
            for( auto &i: clist_rbc[ clist_rbc.p2c(center) ] ) {
            	if (colliding) break;
            	colliding |= collides( t, result_rbc[i] );
            }
            for( auto &i: clist_ctc[ clist_ctc.p2c(center) ] ) {
            	if (colliding) break;
            	colliding |= collides( t, result_ctc[i] );
            }

            if( !colliding ) {
            	clist_rbc.put( t.xmin, t.xmax, result_rbc.size() );
            	result_rbc.push_back( t );
                break;
            }
        } while( ++attempt < maxattempts );

        printf( "attempts: %d, result: %d\n", attempt, result_rbc.size() + result_ctc.size() );

        failed |= attempt == maxattempts;
    }

    double t2 = omp_get_wtime();

    {
    	FILE * f = fopen( "rbcs-ic.txt", "w" );
    	for( auto const &it: result_rbc ) {
    		// COM
            for( int c = 0; c < 3; ++c ) fprintf( f, "%lf ", 0.5 * ( it.xmin[c] + it.xmax[c] ) );
            // rotate matrix
            for( int i = 0; i < 4; ++i ) for( int j = 0; j < 4; ++j ) fprintf( f, "%lf ", it.transform(i,j) );
            fprintf( f, "\n" );
        }
        fclose( f );
    }

    {
    	FILE * f = fopen( "ctcs-ic.txt", "w" );
    	for( auto const &it: result_ctc ) {
            for( int c = 0; c < 3; ++c ) fprintf( f, "%lf ", 0.5 * ( it.xmin[c] + it.xmax[c] ) );
            for( int i = 0; i < 4; ++i ) for( int j = 0; j < 4; ++j ) fprintf( f, "%lf ", it.transform(i,j) );
            fprintf( f, "\n" );
        }
        fclose( f );
    }

	verify( "rbcs-ic.txt" );
	verify( "ctcs-ic.txt" );

	printf("Generation took %lf seconds\n", t2 - t1 );

    return 0;
}
