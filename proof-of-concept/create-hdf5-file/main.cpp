/*
 *  main.cpp
 *  Part of CTC/create-hdf5-file/
 *
 *  Created and authored by Diego Rossinelli on 2015-01-27.
 *  Copyright 2015. All rights reserved.
 *
 *  Users are NOT authorized
 *  to employ the present software for their own publications
 *  before getting a written permission from the author of this file.
 */

#include <mpi.h>
#include <hdf5.h>

#include <cstdlib>
#include <cmath>
#include <cassert>
#include <string>

using namespace std;

const int L = 48;

#define MPI_CHECK(ans) do { mpiAssert((ans), __FILE__, __LINE__); } while(0)

inline void mpiAssert(int code, const char *file, int line, bool abort=true)
{
    if (code != MPI_SUCCESS) 
    {
	char error_string[2048];
	int length_of_error_string = sizeof(error_string);
	MPI_Error_string(code, error_string, &length_of_error_string);
	 
	printf("mpiAssert: %s %d %s\n", file, line, error_string);
	 	 
	MPI_Abort(MPI_COMM_WORLD, code);
    }
}

void write_hdf5field(const char * path, const float *const data, const int nchannels, MPI_Comm cartcomm)
{
    int nranks[3], periods[3], myrank[3];
    MPI_CHECK( MPI_Cart_get(cartcomm, 3, nranks, periods, myrank) );

    id_t plist_id_access = H5Pcreate(H5P_FILE_ACCESS);
    H5Pset_fapl_mpio(plist_id_access, cartcomm, MPI_INFO_NULL);

    hid_t file_id = H5Fcreate(path, H5F_ACC_TRUNC, H5P_DEFAULT, plist_id_access);
    H5Pclose(plist_id_access);
   
    hsize_t globalsize[4] = {nranks[2] * L, nranks[1] * L, nranks[0] * L, 1};    
    hid_t filespace_simple = H5Screate_simple(4, globalsize, NULL);

    char datasetnames[nchannels][512];
    for(int ichannel = 0; ichannel < nchannels; ++ichannel)
	sprintf(datasetnames[ichannel], "data-%d", ichannel);

    for(int ichannel = 0; ichannel < nchannels; ++ichannel)
    {
	hid_t dset_id = H5Dcreate(file_id, datasetnames[ichannel], H5T_NATIVE_FLOAT, filespace_simple, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
	id_t plist_id = H5Pcreate(H5P_DATASET_XFER);

	H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_COLLECTIVE);
	
	hsize_t start[4] = { myrank[2] * L, myrank[1] * L, myrank[0] * L, 0};
	hsize_t extent[4] = { L, L, L,  1};
	hid_t filespace = H5Dget_space(dset_id);
	H5Sselect_hyperslab(filespace, H5S_SELECT_SET, start, NULL, extent, NULL);
	
	hid_t memspace = H5Screate_simple(4, extent, NULL); 
	herr_t status = H5Dwrite(dset_id, H5T_NATIVE_FLOAT, memspace, filespace, plist_id, data + L * L * L * ichannel);
	
	H5Sclose(memspace);
	H5Sclose(filespace);
	H5Pclose(plist_id);		
	H5Dclose(dset_id);
    }

    H5Sclose(filespace_simple);
    H5Fclose(file_id);

    int rankscalar;
    MPI_CHECK(MPI_Comm_rank(cartcomm, &rankscalar));

    if (!rankscalar)
    {
	char wrapper[256];
	sprintf(wrapper, "%s.xmf", path);

	FILE * xmf = fopen(wrapper, "w");
	assert(xmf);
	fprintf(xmf, "<?xml version=\"1.0\" ?>\n");
	fprintf(xmf, "<!DOCTYPE Xdmf SYSTEM \"Xdmf.dtd\" []>\n");
	fprintf(xmf, "<Xdmf Version=\"2.0\">\n");
	fprintf(xmf, " <Domain>\n");
	fprintf(xmf, "   <Grid GridType=\"Uniform\">\n");
	fprintf(xmf, "     <Time Value=\"%05d\"/>\n", 0);
	fprintf(xmf, "     <Topology TopologyType=\"3DCORECTMesh\" Dimensions=\"%d %d %d\"/>\n", 
		(int)globalsize[0], (int)globalsize[1], (int)globalsize[2]);

	fprintf(xmf, "     <Geometry GeometryType=\"ORIGIN_DXDYDZ\">\n");
	fprintf(xmf, "       <DataItem Name=\"Origin\" Dimensions=\"3\" NumberType=\"Float\" Precision=\"4\" Format=\"XML\">\n");
	fprintf(xmf, "        %e %e %e\n", 0.,0.,0.);
	fprintf(xmf, "       </DataItem>\n");
	fprintf(xmf, "       <DataItem Name=\"Spacing\" Dimensions=\"3\" NumberType=\"Float\" Precision=\"4\" Format=\"XML\">\n");

	const float h = 1;
	fprintf(xmf, "        %e %e %e\n", h, h, h);
	fprintf(xmf, "       </DataItem>\n");
	fprintf(xmf, "     </Geometry>\n");

	for(int ichannel = 0; ichannel < nchannels; ++ichannel)
	{
	    fprintf(xmf, "     <Attribute Name=\"Velocity-%d\" AttributeType=\"Scalar\" Center=\"Node\">\n", ichannel/*, datasetnames[ichannel]*/);    
	    fprintf(xmf, "       <DataItem Dimensions=\"%d %d %d\" NumberType=\"Float\" Precision=\"4\" Format=\"HDF\">\n", 
		    (int)globalsize[0], (int)globalsize[1], (int)globalsize[2]);

	    string str(path); 
	    unsigned found = str.find_last_of("/");
	    str = str.substr(found+1);
	    fprintf(xmf, "        %s:/%s\n", str.c_str(), datasetnames[ichannel]);
	
	    fprintf(xmf, "       </DataItem>\n");
	    fprintf(xmf, "     </Attribute>\n");	
	}
	
	fprintf(xmf, "   </Grid>\n");
	fprintf(xmf, " </Domain>\n");
	fprintf(xmf, "</Xdmf>\n");
	fclose(xmf);
    }
}

int main(int argc, char ** argv)
{ 
    int ranks[3];
    
    if (argc != 4)
    {
	printf("usage: ./mpi-dpd <xranks> <yranks> <zranks>\n");
	exit(-1);
    }
    else
    	for(int i = 0; i < 3; ++i)
	    ranks[i] = atoi(argv[1 + i]);

    MPI_CHECK( MPI_Init(&argc, &argv) );
    
    MPI_Comm cartcomm;
    int periods[] = {1, 1, 1};	    
    MPI_CHECK( MPI_Cart_create(MPI_COMM_WORLD, 3, ranks, periods, 1, &cartcomm) );

    int nranks[3],  myrank[3];
    MPI_CHECK( MPI_Cart_get(cartcomm, 3, nranks, periods, myrank) );

    static const int nchannels = 3;
    float * data = new float[L * L * L * nchannels];

    printf("myrank is %d %d %d\n", myrank[0], myrank[1], myrank[2]);

    for(int c = 0; c < nchannels; ++c)
	for(int iz = 0; iz < L; ++iz)
	    for(int iy = 0; iy < L; ++iy)
		for(int ix = 0; ix < L; ++ix)
		{
		    const int gx = ix + myrank[0] * L;
		    const int gy = iy + myrank[1] * L;
		    const int gz = iz + myrank[2] * L;

		    const float x0 = -L * 0.5 * nranks[0];
		    const float y0 = -L * 0.5 * nranks[1];
		    const float z0 = -L * 0.5 * nranks[2];
		    
		    const float p[] = { x0 + gx, y0 + gy, z0 + gz };
		    float v[] = { 0, -p[2], p[1]};
		    
		    const float IvI = sqrtf(v[0]*v[0] + v[1] * v[1] + v[2] * v[2]);
		    v[0] /= IvI;
		    v[1] /= IvI;
		    v[2] /= IvI;

		    data[ix + L * (iy + L * (iz + L * c))] = v[c];
		}

    write_hdf5field("test.h5", data, nchannels, cartcomm);

    delete [] data;

    MPI_CHECK( MPI_Finalize() );
}
 
