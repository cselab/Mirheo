#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <cmath>

int main(const int argc, const char * argv[])
{
    if (argc != 5)
    {
        printf("usage: ./plates <NX> <NY> <NZ> <zmargin>\n");
        return 1;
    }
    
    const int NX = atoi(argv[1]);
    const int NY = atoi(argv[2]);
    const int NZ = atoi(argv[3]);
    const int zmargin = atoi(argv[4]);
    
    float * data = new float[NX * NY * NZ];
    
    for(int iz = 0; iz < NZ; ++iz)
    {
	const float zval = fabs(iz + 0.5 - NZ * 0.5) - (NZ / 2 - zmargin);
	
	for(int iy = 0; iy < NY; ++iy)
	    for(int ix = 0; ix < NX; ++ix)
		data[ix + NX * (iy + NY * iz)] = zval;
    }
        
    FILE * f = fopen("sdf.dat", "w");
    assert(f != 0);
    fprintf(f, "%f %f %f\n", (float)NX, (float)NY, (float)NZ); 
    fprintf(f, "%d %d %d\n", NY, NX, NZ);
    fwrite(data, sizeof(float), NX * NY * NZ, f);
    fclose(f);

#ifndef NDEBUG
    {
	FILE * f = fopen("sdf.raw", "w");
	assert(f != 0);
	fwrite(data, sizeof(float), NX * NY * NZ, f);
	fclose(f);
    }
#endif
    
    delete [] data;
    
    return 0;
}
