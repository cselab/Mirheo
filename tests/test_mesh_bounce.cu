#include <core/rbc_kernels/bounce.h>


int main()
{
	srand48(4);

	for (int tri=0; tri < 10; tri++)
	{
		Particle v0, v1, v2;
		const float dt = 0.1;

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

		expandBy(v0.r, v1.r, v2.r, 0.1);
		n = cross(v1.r-v0.r, v2.r-v0.r);
		ln = dot(n, n);
		n /= sqrt(ln);
		center = 0.333333333f*(v0.r+v1.r+v2.r);

		printf("Expanded tri: [%f %f %f]  [%f %f %f]  [%f %f %f]\n",
				v0.r.x, v0.r.y, v0.r.z, v1.r.x, v1.r.y, v1.r.z, v2.r.x, v2.r.y, v2.r.z);
		printf("Center: [%f %f %f],  normal [%f %f %f],  area %f\n\n\n", center.x, center.y, center.z, n.x, n.y, n.z, 0.5f*ln);


		int N = 100000;
		for (int i=0; i<N;)
		{
			Particle p;
			p.r.x = drand48();
			p.r.y = drand48();
			p.r.z = drand48();

			p.u.x = drand48();
			p.u.y = drand48();
			p.u.z = drand48();

			float oldSign;
			float3 oldCoo = p.r - dt*p.u;
			float3 baricentric = intersectParticleTriangleBaricentric(v0, v1, v2, n, p, dt, 2e-6, oldSign);

			i++;
			float3 real = baricentric.x*v0.r + baricentric.y*v1.r + baricentric.z*v2.r;
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


			if ( dot(p.r-v0.r, n) * dot(oldCoo-r0, cross(r1-r0, r2-r0)) < 0 && baricentric.x < -900 )
			{
				printf("1. [%f %f %f]  -->  [%f %f %f] (%f -> %f):   [%f %f %f] ==> [%f %f %f] (%f)\n",
						oldCoo.x, oldCoo.y, oldCoo.z, p.r.x, p.r.y, p.r.z,
						dot(oldCoo-v0.r, n), dot(p.r-v0.r, n),
						baricentric.x, baricentric.y, baricentric.z,
						real.x, real.y, real.z, v);
				continue;
			}

			if ( (a1+a2+a3) - atot > 1e-4 && (baricentric.x >= 0 && baricentric.y >= 0 && baricentric.z >= 0) )
			{
				printf("2. %f + %f + %f = %f  vs  %f\n", a1, a2, a3, a1+a2+a3, atot);
				printf("2. [%f %f %f]  -->  [%f %f %f] (%f -> %f):   [%f %f %f] ==> [%f %f %f] (%f)\n",
						oldCoo.x, oldCoo.y, oldCoo.z, p.r.x, p.r.y, p.r.z,
						dot(oldCoo-v0.r, n), dot(p.r-v0.r, n),
						baricentric.x, baricentric.y, baricentric.z,
						real.x, real.y, real.z, v);
				continue;
			}


			if (baricentric.x < -900)
				real = p.r;

			real = real + 2e-6f * n * ((oldSign > 0) ? 2.0f : -2.0f);


			if ( dot(oldCoo - r0, cross(r1-r0, r2-r0)) * dot(real-v0.r, n) < 0.0f )
			{

				printf("3. [%f %f %f]  -->  [%f %f %f] (%f -> %f):   [%f %f %f] ==> [%f %f %f] (%f,  %f)\n",
						oldCoo.x, oldCoo.y, oldCoo.z, p.r.x, p.r.y, p.r.z,
						dot(oldCoo-v0.r, n), dot(p.r-v0.r, n),
						baricentric.x, baricentric.y, baricentric.z,
						real.x, real.y, real.z, oldSign, dot(real-v0.r, n));

				continue;
			}
		}

		N = 100000;
		for (int i=0; i<N;)
		{
			Particle p;
			p.r.x = drand48();
			p.r.y = drand48();
			p.r.z = drand48();

			p.u.x = drand48();
			p.u.y = drand48();
			p.u.z = drand48();

			float oldSign;

			float3 oldCoo = p.r - dt*p.u;
			float3 baricentric = intersectParticleTriangleBaricentric(v0, v1, v2, n, p, dt, 2e-6, oldSign);

			if (baricentric.x >= 0.0f && baricentric.y >= 0.0f && baricentric.z >= 0.0f)
			{
				i++;

				float3 real = baricentric.x*v0.r + baricentric.y*v1.r + baricentric.z*v2.r;
				float3 vtri = baricentric.x*v0.u + baricentric.y*v1.u + baricentric.z*v2.u;


				float3 f0 = make_float3(0.0f), f1 = make_float3(0.0f), f2 = make_float3(0.0f);

				float3 U = p.u - vtri;
				triangleForces(v0.r, v1.r, v2.r, 1.0f, baricentric, p.u - vtri, 1.0f, dt, f0, f1, f2);

				float3 mom0 = p.u + v0.u + v1.u + v2.u;
				float3 mom  = p.u - 2*U + v0.u + v1.u + v2.u + (f0 + f1 + f2)*dt;

				float3 l0  = cross(real, p.u) + cross(v0.r, v0.u) + cross(v1.r, v1.u) + cross(v2.r, v2.u);
				float3 l = cross(real, p.u - 2*U) + cross(v0.r, v0.u + f0*dt) + cross(v1.r, v1.u + f1*dt) + cross(v2.r, v2.u + f2*dt);

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
