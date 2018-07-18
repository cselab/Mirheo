#include <core/utils/cuda_common.h>
#include <core/datatypes.h>

#undef  __device__
#define __device__
#define __BOUNCE_TEST__

#include <core/rbc_kernels/bounce.h>


int main()
{
	srand48(4);

	for (int tri=0; tri < 100; tri++)
	{
		Particle v0, v1, v2;
		const float dt = 0.01;

		v0.r.x = drand48();
		v0.r.y = drand48();
		v0.r.z = drand48();
		v1.r.x = drand48();
		v1.r.y = drand48();
		v1.r.z = drand48();
		v2.r.x = drand48();
		v2.r.y = drand48();
		v2.r.z = drand48();

		v0.u.x = 0.1*drand48();
		v0.u.y = 0.1*drand48();
		v0.u.z = 0.1*drand48();
		v1.u.x = 0.1*drand48();
		v1.u.y = 0.1*drand48();
		v1.u.z = 0.1*drand48();
		v2.u.x = 0.1*drand48();
		v2.u.y = 0.1*drand48();
		v2.u.z = 0.1*drand48();

		float3 n = cross(v1.r-v0.r, v2.r-v0.r);
		float ln = dot(n, n);
		n /= sqrt(ln);
		float3 center = 0.333333333f*(v0.r+v1.r+v2.r);

		printf("Initial tri:  [%f %f %f]  [%f %f %f]  [%f %f %f]\n",
				v0.r.x, v0.r.y, v0.r.z, v1.r.x, v1.r.y, v1.r.z, v2.r.x, v2.r.y, v2.r.z);
		printf("Center: [%f %f %f],  normal [%f %f %f],  area %f\n\n", center.x, center.y, center.z, n.x, n.y, n.z, 0.5f*ln);


		int N = 10000;
		for (int i=0; i<N;)
		{
			Particle p, pOld;
			p.r.x = drand48();
			p.r.y = drand48();
			p.r.z = drand48();

			p.u.x = drand48();
			p.u.y = drand48();
			p.u.z = drand48();

			float oldSign;
			pOld.r = p.r - dt*p.u;
			pOld.u = p.u;
			Triangle tr {v0.r, v1.r, v2.r};
			Triangle trOld {v0.r - v0.u*dt, v1.r - v1.u*dt, v2.r - v2.u*dt};

			float3 barycentric = f4tof3( intersectParticleTriangleBarycentric( tr, trOld, p, pOld, oldSign) );

			i++;
			float3 real = barycentric.x*v0.r + barycentric.y*v1.r + barycentric.z*v2.r;
			const float v = dot(real-v0.r, n);

			auto len = [] (float3 v) { return sqrt(dot(v, v)); };
			auto area = [len] (float3 x0, float3 x1, float3 x2) { return len(cross(x1-x0, x2-x0)); };

			float atot = area(v0.r, v1.r, v2.r);
			float a1 = area(real, v0.r, v1.r);
			float a2 = area(real, v1.r, v2.r);
			float a3 = area(real, v2.r, v0.r);

			float3 r0 = v0.r - v0.u*dt;
			float3 r1 = v1.r - v1.u*dt;
			float3 r2 = v2.r - v2.u*dt;


			if ( dot(p.r-v0.r, n) * dot(pOld.r-r0, cross(r1-r0, r2-r0)) < 0 && barycentric.x < -900 )
			{
				printf("1. [%f %f %f]  -->  [%f %f %f] (%f -> %f):   [%f %f %f] ==> [%f %f %f] (%f)\n",
						pOld.r.x, pOld.r.y, pOld.r.z, p.r.x, p.r.y, p.r.z,
						dot(pOld.r-v0.r, n), dot(p.r-v0.r, n),
						barycentric.x, barycentric.y, barycentric.z,
						real.x, real.y, real.z, v);
				continue;
			}

			if ( (a1+a2+a3) - atot > 1e-4 && (barycentric.x >= 0 && barycentric.y >= 0 && barycentric.z >= 0) )
			{
				printf("2. %f + %f + %f = %f  vs  %f\n", a1, a2, a3, a1+a2+a3, atot);
				printf("2. [%f %f %f]  -->  [%f %f %f] (%f -> %f):   [%f %f %f] ==> [%f %f %f] (%f)\n",
						pOld.r.x, pOld.r.y, pOld.r.z, p.r.x, p.r.y, p.r.z,
						dot(pOld.r-v0.r, n), dot(p.r-v0.r, n),
						barycentric.x, barycentric.y, barycentric.z,
						real.x, real.y, real.z, v);
				continue;
			}


			if (barycentric.x < -900)
				real = p.r;

			real = real + 2e-6f * n * ((oldSign > 0) ? 2.0f : -2.0f);


			if ( dot(pOld.r - r0, cross(r1-r0, r2-r0)) * dot(real-v0.r, n) < 0.0f )
			{

				printf("3. [%f %f %f]  -->  [%f %f %f] (%f -> %f):   [%f %f %f] ==> [%f %f %f] (%f,  %f)\n",
						pOld.r.x, pOld.r.y, pOld.r.z, p.r.x, p.r.y, p.r.z,
						dot(pOld.r-v0.r, n), dot(p.r-v0.r, n),
						barycentric.x, barycentric.y, barycentric.z,
						real.x, real.y, real.z, oldSign, dot(real-v0.r, n));

				continue;
			}
		}

		N = 10000;
		for (int i=0; i<N;)
		{
			Particle p, pOld;
			p.r.x = drand48();
			p.r.y = drand48();
			p.r.z = drand48();

			p.u.x = drand48();
			p.u.y = drand48();
			p.u.z = drand48();

			float oldSign;
			pOld.r = p.r - dt*p.u;
			pOld.u = p.u;
			Triangle tr {v0.r, v1.r, v2.r};
			Triangle trOld {v0.r - v0.u*dt, v1.r - v1.u*dt, v2.r - v2.u*dt};

			float3 barycentric = f4tof3( intersectParticleTriangleBarycentric( tr, trOld, p, pOld, oldSign) );

			if (barycentric.x >= 0.0f && barycentric.y >= 0.0f && barycentric.z >= 0.0f)
			{
				i++;

				float3 real = barycentric.x*v0.r + barycentric.y*v1.r + barycentric.z*v2.r;
				float3 vtri = barycentric.x*v0.u + barycentric.y*v1.u + barycentric.z*v2.u;


				float3 f0 = make_float3(0.0f), f1 = make_float3(0.0f), f2 = make_float3(0.0f);

				float3 U = p.u - vtri;
				float3 Unew = make_float3(drand48(), drand48(), drand48());
				float mTri = drand48();
				float mP = drand48();

				triangleForces(tr, mTri, barycentric, U, Unew, mP, dt, f0, f1, f2);

				float3 a0 = f0/mTri, a1 = f1/mTri, a2 = f2/mTri;

				float3 mom0 = mP*p.u + mTri*(v0.u + v1.u + v2.u);
				float3 mom  = mP*(Unew+vtri) + mTri*( v0.u + v1.u + v2.u + (a0 + a1 + a2)*dt );

				float3 l0 = mP*cross(real, p.u) + mTri*( cross(v0.r, v0.u) + cross(v1.r, v1.u) + cross(v2.r, v2.u) );
				float3 l = mP*cross(real, Unew+vtri) + mTri*( cross(v0.r, v0.u + a0*dt) + cross(v1.r, v1.u + a1*dt) + cross(v2.r, v2.u + a2*dt) );

				if (dot(mom - mom0, mom - mom0) > 1e-4 && dot(l0 - l, l0 - l) > 1e-4)
				{
					printf("M:  [%f %f %f]  -->  [%f %f %f]\n", mom0.x, mom0.y, mom0.z, mom.x, mom.y, mom.z);
					printf("L:  [%f %f %f]  -->  [%f %f %f]\n\n", l0.x, l0.y, l0.z, l.x, l.y, l.z);
				}
			}
		}
	}

	return 0;
}
