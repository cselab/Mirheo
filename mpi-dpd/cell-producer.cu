/*
 *  cell-producer.cpp
 *  ctc-garbald
 *
 *  Created by Dmitry Alexeev on Mar 21, 2015
 *  Copyright 2015 ETH Zurich. All rights reserved.
 *
 */


#include "cell-producer.h"

CellProducer::CellProducer(int n, int m, float xmin, float xmax, int xcoo): n(n), m(m), xmin(xmin), xmax(xmax), xcoo(xcoo)
{
	occupied = new int*[n];
	for (int j=0; j<n; j++)
		occupied[j] = new int[m];

	for (int j=0; j<n; j++)
		for (int k=0; k<n; k++)
			occupied[j][k] = 0;

	hy = YSIZE_SUBDOMAIN / n;
	hz = ZSIZE_SUBDOMAIN / m;

	rng = Logistic::KISS(1234, 6978, 290, 12968);
}

/* Ken Shoemake, September 1991 */
struct Quat
{
	float x, y, z, w;
};

/** Qt_Random
 *  Generate uniform random unit quaternion from uniform deviates.
 *  Each x[i] should vary between 0 and 1.
 */
inline Quat qtRandom(float x0, float x1, float x2)
{
	/*  The subgroup algorithm can be condensed to this efficient form.
	 *  Use rotations around z as a subgroup, with coset representatives
	 *  the rotations pointing the z axis in different directions.
	 */
	Quat q;
	float r1 = sqrt(1.0 - x0), r2 = sqrt(x0);
	float t1 = 2*M_PI*x1, t2 = 2*M_PI*x2;
	float c1 = cos(t1), s1 = sin(t1);
	float c2 = cos(t2), s2 = sin(t2);
	q.x = s1*r1; q.y = c1*r1; q.z = s2*r2; q.w = c2*r2;
	return q;
}

/** Qt_ToMatrix
 *  Construct rotation matrix from quaternion (unit or not).
 *  Assumes matrix is used to multiply row vector on the right:
 *  vnew = vold mat.  Works correctly for right-handed coordinate system
 *  and right-handed rotations. For column vectors or for left-handed
 *  coordinate systems, transpose the matrix.
 */
inline void qt2mat(Quat q,  float (*m)[4])
{
	float norm = q.x*q.x + q.y*q.y + q.z*q.z + q.w*q.w;
	float s = (norm > 0.0) ? 2.0/norm : 0.0;
	float xs = q.x*s,	ys = q.y*s,		zs = q.z*s;
	float wx = q.w*xs,	wy = q.w*ys,	wz = q.w*zs,
			xx = q.x*xs,	xy = q.x*ys,	xz = q.x*zs,
			yy = q.y*ys,	yz = q.y*zs,	zz = q.z*zs;

	m[0][0] = 1.0 - (yy + zz); m[0][1] = xy + wz;         m[0][2] = xz - wy;
	m[1][0] = xy - wz;         m[1][1] = 1.0 - (xx + zz); m[1][2] = yz + wx;
	m[2][0] = xz + wy;         m[2][1] = yz - wx;         m[2][2] = 1.0 - (xx + yy);
}

inline float CellProducer::toGlobal(float x)
{
	return (xcoo+0.5)*XSIZE_SUBDOMAIN + x;
}

bool CellProducer::produce(ParticleArray& p, CollectionRBC* rbcs)
{
	if (rbcs == NULL) return false;
	const int attempts = 10000;

	//printf("Generating a cell NOW!\n");

	PinnedHostBuffer<CudaRBC::Extent> extents(rbcs->count() + 1);
	CudaRBC::Extent* orig = extents.data + rbcs->count();
	CudaRBC::Extent* cells = extents.data;

	CudaRBC::extent_nohost(0, rbcs->count(), (float*)rbcs->data(), extents.devptr);
	CudaRBC::extent_nohost(0, 1, CudaRBC::get_orig_xyzuvw(), extents.devptr + rbcs->count());
	cudaDeviceSynchronize();

	for (int j=0; j<n; j++)
		for (int k=0; k<m; k++)
			occupied[j][k] = 0;

	for (int i=0; i<rbcs->count(); i++)
	{
		int jmin = max((cells[i].ymin + YSIZE_SUBDOMAIN * 0.5f) / hy, (float)0);
		int jmax = min((cells[i].ymax + YSIZE_SUBDOMAIN * 0.5f) / hy, (float)n-1);
		int kmin = max((cells[i].zmin + ZSIZE_SUBDOMAIN * 0.5f) / hz, (float)0);
		int kmax = min((cells[i].zmax + ZSIZE_SUBDOMAIN * 0.5f) / hz, (float)m-1);

		//		printf("x: %f %f;  %f  %f\n", cells[i].xmin, cells[i].xmax, xmin, xmax);
		//		printf("y: %f %f => %d %d\n", cells[i].ymin, cells[i].ymax, jmin, jmax);
		//		printf("z: %f %f => %d %d\n", cells[i].zmin, cells[i].zmax, kmin, kmax);

		for (int j=jmin; j<=jmax; j++)
			for (int k=kmin; k<=kmax; k++)
				if (toGlobal(cells[i].xmin) < xmax) occupied[j][k] = 1;
	}

	//	printf("Grid looks as follows:\n");
	//	for (int j=0; j<n; j++)
	//	{
	//		for (int k=0; k<m; k++)
	//			printf("%1d ", occupied[j][k]);
	//		printf("\n");
	//	}

	int free = 0;
	for (int j=0; j<n; j++)
		for (int k=0; k<m; k++)
			free += 1 - occupied[j][k];

	if (free == 0) return false;

	int pos = rng.get_int() % free;
	int cur = 0, myj, myk;
	for (int j=0; j<n; j++)
		for (int k=0; k<m; k++)
		{
			if (pos == cur)
			{
				myj = j;
				myk = k;
			}
			cur += 1 - occupied[j][k];
		}

	printf("Will insert into cell %d => [%d  %d] out of %d free cells\n", pos, free, myj, myk);

	float A[4][4];
	memset(A[0], 0, 4*sizeof(float));
	memset(A[1], 0, 4*sizeof(float));
	memset(A[2], 0, 4*sizeof(float));
	memset(A[3], 0, 4*sizeof(float));

	Quat q = qtRandom(rng.get_float(), rng.get_float(), rng.get_float());
	qt2mat(q, A);

	float maxLen = max(max(orig->xmax - orig->xmin, orig->ymax - orig->ymin), orig->zmax - orig->zmin);
	if (maxLen > hy || maxLen > hz)
	{
		return false;
	}

	float xshift = maxLen*0.5 + (xmin - (xcoo+0.5)*XSIZE_SUBDOMAIN) + ((xmax - xmin) - maxLen) * rng.get_float();
	float yshift = maxLen*0.5 + myj * hy - YSIZE_SUBDOMAIN*0.5 + (hy - maxLen) * rng.get_float();
	float zshift = maxLen*0.5 + myk * hz - ZSIZE_SUBDOMAIN*0.5 + (hz - maxLen) * rng.get_float();

	// Shift A
	A[0][3] = xshift;
	A[1][3] = yshift;
	A[2][3] = zshift;
	A[3][3] = 1;

	printf("Shifting by [%5f  %5f  %5f]\n", xshift, yshift, zshift);

	rbcs->preserve_resize(rbcs->count() + 1);
	extents.resize(rbcs->count() + 1);
	CudaRBC::initialize((float*)(rbcs->data() + (rbcs->count() - 1)*rbcs->nvertices), A);

//	CudaRBC::extent_nohost(0, rbcs->count(), (float*)rbcs->data(), extents.devptr);
//	cudaDeviceSynchronize();
//
//	//if (rbcs->count() == 2)
//	{
//		float *xy = new float[6*500];
//		cudaMemcpy(xy, rbcs->data(), 6*500*sizeof(float), cudaMemcpyDeviceToHost);
//
//		for (int i=0; i<500; i++)
//			printf("%d:  %f   %f   %f\n", i, xy[6*i + 0], xy[6*i + 1], xy[6*i + 2]);
//	}
//
//	for (int i=0; i<rbcs->count() + 1; i++)
//		printf("cell %d extent:  %f %f %f   %f %f %f\n", i,
//				extents.data[i].xmin, extents.data[i].ymin, extents.data[i].zmin,
//				extents.data[i].xmax, extents.data[i].ymax, extents.data[i].zmax);

	return true;
}
