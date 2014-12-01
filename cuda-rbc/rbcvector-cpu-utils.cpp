/*
 *  rbcvector.cpp
 *  ctc phenix
 *
 *  Created by Dmitry Alexeev on Nov 17, 2014
 *  Copyright 2014 ETH Zurich. All rights reserved.
 *
 */

#include <cassert>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <cstring>

#include "rbcvector.h"
#include "rbcvector-cpu-utils.h"

using namespace std;

void cpu_loadHeader(RBCVector& rbcs, const char* fname, bool report)
{
	ifstream in(fname);
	string line;

	rbcs.rbcdiam = 11;
	rbcs.nparticles = -1;
	rbcs.ntriang = -1;
	rbcs.nbonds = -1;
	rbcs.ndihedrals = -1;
	int ntot = -1;

	if (report)
		if (in.good())
		{
			cout << "Reading file " << fname << endl;
		}
		else
		{
			cout << fname << ": no such file" << endl;
			exit(1);
		}

	in >> ntot >> rbcs.nbonds >> rbcs.ntriang >> rbcs.ndihedrals;

	if (report)
		if (in.good())
		{
			cout << "File contains " << ntot << " atoms, " << rbcs.nbonds << " bonds, " << rbcs.ntriang << " triangles and " << rbcs.ndihedrals << " dihedrals" << endl;
		}
		else
		{
			cout << "Couldn't parse the file" << endl;
			exit(1);
		}

	// Atoms section
	real *xyzuvwtmp = new real[6*ntot];

	int cur = 0;
	int tmp1, tmp2, aid, mid;
	while (in.good() && cur < ntot)
	{
		in >> tmp1 >> tmp2 >> aid >> xyzuvwtmp[6*cur+0] >> xyzuvwtmp[6*cur+1] >> xyzuvwtmp[6*cur+2];
		xyzuvwtmp[6*cur+3] = xyzuvwtmp[6*cur+4] = xyzuvwtmp[6*cur+5] = 0;
		if (aid != 1) break;
		cur++;
	}

	// Shift the origin of "zeroth" rbc to 0,0,0
	float com[3] = {0, 0, 0};
	for (int i=0; i<cur; i++)
		for (int d=0; d<3; d++)
			com[d] += xyzuvwtmp[6*i + d];

	for (int d=0; d<3; d++)
		com[d] /= cur;

	for (int i=0; i<cur; i++)
		for (int d=0; d<3; d++)
			xyzuvwtmp[6*i + d] -= com[d];


	rbcs.nparticles = cur;
	rbcs.orig_xyzuvw = new real[6*rbcs.nparticles];
	//uvw = new real[n];

	memcpy(rbcs.orig_xyzuvw, xyzuvwtmp, 6*rbcs.nparticles*sizeof(real));

	int *used = new int[rbcs.nparticles];
	rbcs.mapping = new int[4*(rbcs.ntriang + rbcs.ndihedrals)];
	memset(used, 0, rbcs.nparticles*sizeof(int));
	memset(rbcs.mapping, 0, 4*(rbcs.ntriang + rbcs.ndihedrals)*sizeof(int));
	int id0, id1, id2, id3;

	// Bonds section

	rbcs.bonds = new int[rbcs.nbonds * 2];
	for (int i=0; i<rbcs.nbonds; i++)
	{
		in >> tmp1 >> tmp2 >> id0 >> id1;
		id0--; id1--;
		rbcs.bonds[2*i + 0] = id0;
		rbcs.bonds[2*i + 1] = id1;
	}

	// Angles section --> triangles

	rbcs.triangles = new int[3*rbcs.ntriang];
	for (int i=0; i<rbcs.ntriang; i++)
	{
		in >> tmp1 >> tmp2 >> id0 >> id1 >> id2;

		id0--; id1--; id2--;
		rbcs.triangles[3*i + 0] = id0;
		rbcs.triangles[3*i + 1] = id1;
		rbcs.triangles[3*i + 2] = id2;

		rbcs.mapping[4*i + 0] = (used[id0]++);
		rbcs.mapping[4*i + 1] = (used[id1]++);
		rbcs.mapping[4*i + 2] = (used[id2]++);
	}

	// Dihedrals section

	rbcs.dihedrals = new int[4*rbcs.ndihedrals];
	for (int i=0; i<rbcs.ndihedrals; i++)
	{
		in >> tmp1 >> tmp2 >> id0 >> id1 >> id2 >> id3;
		id0--; id1--; id2--; id3--;

		rbcs.dihedrals[4*i + 0] = id0;
		rbcs.dihedrals[4*i + 1] = id1;
		rbcs.dihedrals[4*i + 2] = id2;
		rbcs.dihedrals[4*i + 3] = id3;

		rbcs.mapping[4*(i + rbcs.ntriang) + 0] = (used[id0]++);
		rbcs.mapping[4*(i + rbcs.ntriang) + 1] = (used[id1]++);
		rbcs.mapping[4*(i + rbcs.ntriang) + 2] = (used[id2]++);
		rbcs.mapping[4*(i + rbcs.ntriang) + 3] = (used[id3]++);
	}

	rbcs.pfxfyfz = NULL;

	in.close();
}

void cpu_initUnique(RBCVector& rbcs, vector<vector<float> > origins, vector<float> coolo, vector<float> coohi)
{
	// Assume that header is already there

	// How many cells does my rank have?

	int mine = 0;
	for (int m = 0; m<origins.size(); m++)
	{
		bool inside = true;
		for (int d=0; d<3; d++)
			inside = inside && (coolo[d] < origins[m][d] && origins[m][d] <= coohi[d]);

		if (inside)
			mine++;
	}

	int cellSize = 6 * rbcs.nparticles;
	rbcs.xyzuvw = new float[cellSize * mine];
	rbcs.n = mine;
	rbcs.maxSize = mine;

	rbcs.ids = new int[mine];

	// Now copy the coordinates and shift them accordingly

	int cur = 0;
	for (int m = 0; m<origins.size(); m++)
	{
		bool inside = true;
		for (int d=0; d<3; d++)
			inside = inside && (coolo[d] < origins[m][d] && origins[m][d] <= coohi[d]);

		if (inside)
		{
			float* curStart = rbcs.xyzuvw + cur*cellSize;
			memcpy(curStart, rbcs.orig_xyzuvw, cellSize * sizeof(float));
			for (int i=0; i<rbcs.nparticles; i++)
				for (int d=0; d<3; d++)
					curStart[6*i + d] += origins[m][d] - coolo[d] - 0.5*(coohi[d] - coolo[d]);

			rbcs.ids[cur] = m;
			cur++;
		}
	}

	rbcs.bounds = new float[6 * rbcs.n];
	rbcs.coms   = new float[3 * rbcs.n];

	// Beware. 28 = 27 + 1, so here i assume no more that 27 ranks
	//  have a cell simultaneously, and 1 extra int (-1) to show
	//  where owners list finishes
	rbcs.owners = new int[28 * rbcs.n];
}

void cpu_boundsAndComs(RBCVector& rbcs)
{
	for (int m = 0; m<rbcs.n; m++)
	{
		float* curStart = rbcs.xyzuvw + m * 6 * rbcs.nparticles;
		for (int d=0; d<3; d++)
		{
			rbcs.bounds[6*m + 0 + d] =  1e10;  // min d-th dimension
			rbcs.bounds[6*m + 3 + d] = -1e10;  // max d-th dimension
			rbcs.coms  [3*m + d]     =  0;
		}

		for (int i=0; i<rbcs.nparticles; i++)
		{
			for (int d=0; d<3; d++)
			{
				rbcs.bounds[6*m + 0 + d] = min(curStart[6*i + d], rbcs.bounds[6*m + 0 + d]);
				rbcs.bounds[6*m + 3 + d] = max(curStart[6*i + d], rbcs.bounds[6*m + 3 + d]);
				rbcs.coms[3*m + d]      += curStart[6*i + d];
			}
		}
		for (int d=0; d<3; d++)
			rbcs.coms[3*m + d] /= rbcs.nparticles;
	}

}


void cpu_reallocate(RBCVector& rbcs, float* &fxfyfz, int nsize)
{
	if (rbcs.maxSize < nsize)
	{
		printf("!!! reallocating %d --> %d\n", rbcs.n, nsize);

		bool todel = (rbcs.maxSize == 0);
		int oldn = rbcs.n;
		rbcs.n = rbcs.maxSize = nsize;

		if (todel)
		{
			delete[] rbcs.bounds;
			delete[] rbcs.coms;
			delete[] rbcs.owners;
		}

		rbcs.bounds = new float[6 * rbcs.n];
		rbcs.coms   = new float[3 * rbcs.n];
		rbcs.owners = new int[28 * rbcs.n];


		float* tmpxyz = new float[6*rbcs.nparticles * rbcs.n];
		int*   tmpids = new int[rbcs.n];
		memcpy(tmpxyz, rbcs.xyzuvw, 6*rbcs.nparticles * oldn * sizeof(float));
		memcpy(tmpids, rbcs.ids,                        oldn * sizeof(int));

		if (todel)
		{
			delete[] rbcs.xyzuvw;
			delete[] rbcs.ids;
		}

		rbcs.xyzuvw = tmpxyz;
		rbcs.ids    = tmpids;

		if (todel)
			delete[] fxfyfz;

		fxfyfz = new float[3*rbcs.nparticles * rbcs.n];
	}
	else
	{
		rbcs.n = nsize;
	}
}




