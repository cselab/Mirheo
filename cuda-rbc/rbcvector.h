/*
 *  rbcvector.h
 *  ctc phenix
 *
 *  Created by Dmitry Alexeev on Nov 17, 2014
 *  Copyright 2014 ETH Zurich. All rights reserved.
 *
 */


#pragma once

#include "misc.h"

struct RBCVector
{
	// Header
	int nparticles;
	int ntriang;
	int nbonds;
	int ndihedrals;

	int *triangles;
	int *bonds;
	int *dihedrals;

	// Helper pointers
	//real *totA_V;
	int  *mapping;

	// Original configuration
	real* orig_xyzuvw;

	float rbcdiam;

	// Cells
	int n;
	int maxSize;
	int*  ids;
	real* xyzuvw;
	real* pfxfyfz;

	float* bounds;  // 0 - xmin, 1 - ymin, 2 - zmin, 3 - xmax, 4 - ymax, 5 - zmax
	float* coms;
	int*   owners;
};



//struct RBCVector
//{
//	static MPI_Datatype myCoordType, myForceType;
//	static bool coordInitalized;
//	static bool forceInitalized;
//
//	static MPI_Datatype coordType()
//	{
//		if (!coordInitalized)
//		{
//			int          blen[2]     = {1, 6*nparticles};
//			MPI_Aint     indices[2]  = {0, sizeof(int)};
//			MPI_Datatype oldtypes[2] = {MPI_INT, MPI_FLOAT};
//
//			MPI_CHECK( MPI_Type_struct(2, blen, indices, oldtypes, &myCoordType ) );
//			MPI_CHECK( MPI_Type_commit(&myCoordType) );
//
//			coordInitalized = true;
//		}
//
//		return myCoordType;
//	}
//
//	static MPI_Datatype forceType()
//	{
//		if (!forceInitalized)
//		{
//			MPI_CHECK( MPI_Type_contiguous(nparticles * 3, MPI_FLOAT, &myForceType) );
//			MPI_CHECK( MPI_Type_commit(&myForceType) );
//
//			forceInitalized = true;
//		}
//
//		return myForceType;
//	}
//
//	int n;
//
//	MPI_Comm cartcomm;
//	int L, myrank, dims[3], periods[3], coords[3], rankneighbors[27];
//
//	Membrane* zeroPatient;
//	vector<Membrane*> rbcs;
//	vector<vector<float> > bounds;  // 0 - xmin, 1 - ymin, 2 - zmin, 3 - xmax, 4 - ymax, 5 - zmax
//	vector<vector<float> > com;
//	vector<float> fxfyfz;
//
//	static int nparticles;
//	float rbcdiam;
//
//	vector<vector<int>> owners;
//
//	RBCVector(MPI_Comm cartcomm, int L);
//	void boundsAndComs();
//	void getOwners();
//	int  idByCom(vector<float>& com);
//	void init(vector<vector<float>> origins);
//	void init_nonlocal(vector<vector<float>> origins);
//};
