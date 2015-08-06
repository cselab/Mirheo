/*
 *  main.cpp
 *  Part of CTC/cell-placement/
 *
 *  Created and authored by Diego Rossinelli on 2014-12-18.
 *  Further edited by Dmitry Alexeev on 2014-03-25.
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
#include <algorithm>
#include <list>

using namespace std;

struct Extent
{
    float xmin, ymin, zmin;
    float xmax, ymax, zmax;
};

Extent compute_extent(const char * const path)
{
    ifstream in(path);
    string line;

    if (in.good())
	cout << "Reading file " << path << endl;
    else
    {
	cout << path << ": no such file" << endl;
	exit(1);
    }

    int nparticles, nbonds, ntriang, ndihedrals;

    in >> nparticles >> nbonds >> ntriang >> ndihedrals;

    if (in.good())
	cout << "File contains " << nparticles << " atoms, " << nbonds << " bonds, " << ntriang
	     << " triangles and " << ndihedrals << " dihedrals" << endl;
    else
    {
	cout << "Couldn't parse the file" << endl;
	exit(1);
    }

    vector<float> xs(nparticles), ys(nparticles), zs(nparticles);

    for(int i = 0; i < nparticles; ++i)
    {
	int dummy;
	in >> dummy >> dummy >> dummy >> xs[i] >> ys[i] >> zs[i];
    }

    Extent retval = {
	*min_element(xs.begin(), xs.end()),
	*min_element(ys.begin(), ys.end()),
	*min_element(zs.begin(), zs.end()),
	*max_element(xs.begin(), xs.end()),
	*max_element(ys.begin(), ys.end()),
	*max_element(zs.begin(), zs.end())
    };

    {
	printf("extent: \n");
	for(int i = 0; i < 6; ++i)
	    printf("%f ", *(i + (float *)(&retval.xmin)));
	printf("\n");
    }

    return retval;
}

struct TransformedExtent
{
    float transform[4][4];

    float xmin[3], xmax[3], local_xmin[3], local_xmax[3];

    TransformedExtent(Extent extent, const int domain_extent[3])
	{
	    local_xmin[0] = extent.xmin;
	    local_xmin[1] = extent.ymin;
	    local_xmin[2] = extent.zmin;

	    local_xmax[0] = extent.xmax;
	    local_xmax[1] = extent.ymax;
	    local_xmax[2] = extent.zmax;

	    build_transform(extent, domain_extent);

	    for(int i = 0; i < 8; ++i)
	    {
		const int idx[3] = { i % 2, (i/2) % 2, (i/4) % 2 };

		float local[3];
		for(int c = 0; c < 3; ++c)
		    local[c] = idx[c] ? local_xmax[c] : local_xmin[c];

		float world[3];

		apply(local, world);

		if (i == 0)
		    for(int c = 0; c < 3; ++c)
			xmin[c] = xmax[c] = world[c];
		else
		    for(int c = 0; c < 3; ++c)
		    {
			xmin[c] = min(xmin[c], world[c]);
			xmax[c] = max(xmax[c], world[c]);
		    }
	    }
	}

    void build_transform(const Extent extent, const int domain_extent[3])
	{
	    for(int i = 0; i < 4; ++i)
		for(int j = 0; j < 4; ++j)
		    transform[i][j] = i == j;

	    for(int i = 0; i < 3; ++i)
		transform[i][3] = - 0.5 * (local_xmin[i] + local_xmax[i]);

	    const float angles[3] = {
		(float)(0.25 * (drand48() - 0.5) * 2 * M_PI),
		(float)(M_PI * 0.5 + 0.25 * (drand48() * 2 - 1) * M_PI),
		(float)(0.25 * (drand48() - 0.5) * 2 * M_PI)
	    };

	    for(int d = 0; d < 3; ++d)
	    {
		const float c = cos(angles[d]);
		const float s = sin(angles[d]);

		float tmp[4][4];

		for(int i = 0; i < 4; ++i)
		    for(int j = 0; j < 4; ++j)
			tmp[i][j] = i == j;

		if (d == 0)
		{
		    tmp[0][0] = tmp[1][1] = c;
		    tmp[0][1] = -(tmp[1][0] = s);
		}
		else
		    if (d == 1)
		    {
			tmp[0][0] = tmp[2][2] = c;
			tmp[0][2] = -(tmp[2][0] = s);
		    }
		    else
		    {
			tmp[1][1] = tmp[2][2] = c;
			tmp[1][2] = -(tmp[2][1] = s);
		    }

		float res[4][4];
		for(int i = 0; i < 4; ++i)
		    for(int j = 0; j < 4; ++j)
		    {
			float s = 0;

			for(int k = 0; k < 4; ++k)
			    s += transform[i][k] * tmp[k][j];

			res[i][j] = s;
		    }

		for(int i = 0; i < 4; ++i)
		    for(int j = 0; j < 4; ++j)
			transform[i][j] = res[i][j];
	    }

	    float maxlocalextent = 0;
	    for(int i = 0; i < 3; ++i)
		maxlocalextent = max(maxlocalextent, local_xmax[i] - local_xmin[i]);

	    for(int i = 0; i < 3; ++i)
		transform[i][3] += 0.5 * maxlocalextent + drand48() * (domain_extent[i] - maxlocalextent);
	}

    void apply(float x[3], float y[3])
	{
	    for(int i = 0; i < 3; ++i)
		y[i] = transform[i][0] * x[0] + transform[i][1] * x[1] + transform[i][2] * x[2] + transform[i][3];
	}

    bool collides(const TransformedExtent a, const float tol)
	{
	    float s[3], e[3];

	    for(int c = 0; c < 3; ++c)
	    {
		s[c] = max(xmin[c], a.xmin[c]);
		e[c] = min(xmax[c], a.xmax[c]);

		if (s[c] - e[c] >= tol)
		    return false;
	    }

	    return true;
	}
};

void verify(string path2ic)
{
    printf("VERIFYING <%s>\n", path2ic.c_str());

    FILE * f = fopen(path2ic.c_str(), "r");

    bool isgood = true;

    while(isgood)
    {
	float tmp[19];
	for(int c = 0; c < 19; ++c)
	{
	    int retval = fscanf(f, "%f", tmp + c);

	    isgood &= retval == 1;
	}

	if (isgood)
	{
	    printf("reading: ");

	    for(int c = 0; c < 19; ++c)
		printf("%f ", tmp[c]);

	    printf("\n");
	}
    }

    fclose(f);

    printf("========================================\n\n\n\n");
}

class Checker
{
    const float safetymargin;
    float h[3];
    int n[3], ntot;

    vector<list<TransformedExtent> > data;

public:

    Checker(float hh, int dext[3], const float safetymargin):
	safetymargin(safetymargin)
	{
	    h[0] = h[1] = h[2] = hh;

	    for (int d = 0; d < 3; ++d)
		n[d] = (int)ceil((double)dext[d] / h[d]) + 2;

	    ntot = n[0] * n[1] * n[2];

	    data.resize(ntot);
	}

    bool check(TransformedExtent& ex)
	{
	    int imin[3], imax[3];

	    for (int d=0; d<3; d++)
	    {
		imin[d] = floor(ex.xmin[d] / h[d]) + 1;
		imax[d] = floor(ex.xmax[d] / h[d]) + 1;
	    }

	    for (int i=imin[0]; i<=imax[0]; i++)
		for (int j=imin[1]; j<=imax[1]; j++)
		    for (int k=imin[2]; k<=imax[2]; k++)
		    {
			const int icell = i * n[1] * n[2] + j * n[2] + k;

			bool good = true;

			for (auto rival : data[icell])
			    good &= !ex.collides(rival, safetymargin);

			if (!good)
			    return false;
		    }

	    return true;
	}


    void add(TransformedExtent& ex)
	{
	    int imin[3], imax[3];

	    for (int d=0; d<3; ++d)
	    {
		imin[d] = floor(ex.xmin[d] / h[d]) + 1;
		imax[d] = floor(ex.xmax[d] / h[d]) + 1;
	    }

	    bool good = true;

	    for (int i=imin[0]; i<=imax[0]; i++)
		for (int j=imin[1]; j<=imax[1]; j++)
		    for (int k=imin[2]; k<=imax[2]; k++)
		    {
			const int icell = i * n[1]*n[2] + j * n[2] + k;
			data[icell].push_back(ex);
		    }
	}
};

int main(int argc, const char ** argv)
{
    if ((argc != 4)&&(argc != 7))
    {
	printf("usage-1: ./cell-placement <xdomain-extent> <ydomain-extent> <zdomain-extent>\n");
	printf("usage-2: ./cell-placement <local_xdomain-extent> <local_ydomain-extent> <local_zdomain-extent> <xranks> <yranks> <zranks>\n");
	exit(-1);
    }

    int domainextent[3];
    if (argc == 4)
    {
	for(int i = 0; i < 3; ++i)
	    domainextent[i] = atoi(argv[1 + i]);
    }
    if (argc == 7)
    {
	int ldomainextent[3];
	int ranki[3];
	for(int i = 0; i < 3; ++i)
	{
	    ldomainextent[i] = atoi(argv[1 + i]);
	    ranki[i] = atoi(argv[4 + i]);
	    domainextent[i] = ldomainextent[i]*ranki[i];
	}
    }

    printf("domain extent: %d %d %d\n",
	   domainextent[0], domainextent[1], domainextent[2]);

    Extent extents[2] = {
	compute_extent("../cuda-rbc/rbc2.atom_parsed"),
	compute_extent("../cuda-ctc/sphere.dat")
    };

    bool failed = false;

    vector<TransformedExtent> results[2];

    const float tol = 0.7;

    Checker checker(8, domainextent, tol);

    int tot = 0;
    while(!failed)
    {
	const int maxattempts = 100000;

	int attempt = 0;
	for(; attempt < maxattempts; ++attempt)
	{
	    const int type = 0;//(int)(drand48() >= 0.25);

	    TransformedExtent t(extents[type], domainextent);

	    bool noncolliding = true;

#if 0
            //original code
	    for(int i = 0; i < 2; ++i)
		for(int j = 0; j < results[i].size() && noncolliding; ++j)
		    noncolliding &= !t.collides(results[i][j], tol);
#else
            noncolliding = checker.check(t);
#endif

            if (noncolliding)
	    {
                checker.add(t);
		results[type].push_back(t);
                ++tot;
		break;
	    }
	}

        if (tot % 1000 == 0)
	    printf("Done with %d cells...\n", tot);

	failed |= attempt == maxattempts;
    }

    string output_names[2] = { "rbcs-ic.txt", "ctcs-ic.txt" };

    for(int idtype = 0; idtype < 2; ++idtype)
    {
	FILE * f = fopen(output_names[idtype].c_str(), "w");

	for(vector<TransformedExtent>::iterator it = results[idtype].begin(); it != results[idtype].end(); ++it)
	{
	    for(int c = 0; c < 3; ++c)
		fprintf(f, "%f ", 0.5 * (it->xmin[c] + it->xmax[c]));

	    for(int i = 0; i < 4; ++i)
		for(int j = 0; j < 4; ++j)
		    fprintf(f, "%f ", it->transform[i][j]);

	    fprintf(f, "\n");
	}

	fclose(f);
    }

    printf("Generated %d RBCs, %d CTCs\n", (int)results[0].size(), (int)results[1].size());

    return 0;
}
