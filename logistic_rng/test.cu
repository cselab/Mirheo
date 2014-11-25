#include <cstdio>
#include <cstdlib>
#include <climits>
#include "logistic.h"

__global__ void generate( float *output, float trunk, int n_particle )
{
    for( int i = blockIdx.x * blockDim.x + threadIdx.x; i < n_particle; i += gridDim.x * blockDim.x ) {
        for( int j = 0; j < n_particle; j++ ) {
            output[ i + j * n_particle ] = logistic<11, float>( trunk, i, j );
        }
    }
}

int main()
{
    int n = 100;
    
    srand( 0 );
    float trunk = double( rand() ) / RAND_MAX;

    float *output;
    cudaMallocHost( &output, n * n * sizeof( float ) );
    
    generate <<< 1, 512>>>( output, trunk, n );
    cudaDeviceSynchronize();
    
    float *p = output;
    for( int i = 0; i < n; i++ ) {
        for( int j = 0; j < n; j++ ) {
            printf( "%f%c ", *p++, ( j == n - 1 ) ? '\n' : ' ' );
        }
    }
}