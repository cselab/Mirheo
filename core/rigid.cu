#include <core/object_vector.h>
#include <core/rigid_object_vector.h>
#include <core/helper_math.h>

__device__ inline float sqr (float x)
{
	return x*x;
}

__device__ inline void bounceParticle(float3& coo, float3& vel, float dt, float3 objCom, float3 objVel, float3 objOmega, float3 axes, float3& objForce, float3& objTorque)
{
	const float3 vshift = objVel + cross(objOmega, coo - objCom);
	const float3 vr = vel - vshift;

	// ===========================

	const float u = vr.x;
	const float v = vr.y;
	const float w = vr.z;

	const float x = coo.x;
	const float y = coo.y;
	const float z = coo.z;

	const float a = axes.x;
	const float b = axes.y;
	const float c = axes.z;

	const float x0 = objCom.x;
	const float y0 = objCom.y;
	const float z0 = objCom.z;

	// ===========================

	const float t3 = x0 - x;
	const float t5 = b * b;
	const float t7 = a * a;
	const float t8 = y0 - y;
	const float t12 = c * c;
	const float t14 = z0 - z;
	const float t18 = w * w;
	const float t19 = t5 * t18;
	const float t20 = v * c;
	const float t21 = -v * t14;
	const float t22 = t8 * w;
	const float t28 = u * c;
	const float t29 = -u * t14;
	const float t30 = t3 * w;
	const float t38 = sqr(t3 * v - u * t8);
	const float t42 = t5 * t12;
	const float t43 = -t42 * t7 * (t7 * (-t19 + (-t20 + t21 + t22) * (t20 + t21 + t22)) + t5 * (t28 + t29 + t30) * (-t28 + t29 + t30) + t38 * t12);

	if (t43 < 0) return;

	const float t44 = sqrtf(t43);
	const float t46 = v * v;
	const float t50 = u * u;

	const float alpha = 0.1e1f / (t7 * (t12 * t46 + t19) + t50 * t42);
	const float beta = t12 * (t5 * u * t3 + v * t8 * t7) + t5 * w * t14 * t7;

	const float res1 = alpha * (beta + t44);
	const float res2 = alpha * (beta - t44);

	// ===========================

	const float t0 = -1;
	if (0 <= res1 && res1 <= dt)
		t0 = res1;
	if (0 <= res2 && res2 <= dt)
		t0 = res2;

	if (t0 < 0) return;

	coo = coo + (t0 - (dt - t0)) * vr;
	vel = -vel + 2*vshift;

	const float3 f = 2*(vshift - vel) * dt;
	objForce += f;
	objTorque += cross(coo - objCom, f);
}

__global__ void bounceFromEllipsoid(float4* coosvels, const ObjectVector::COMandExtent* props, RigidObjectVector::RigidMovement* vfts, const int nObj, const float3 axes, const float r,
		const uint* __restrict__ cellsStartSize, CellListInfo cinfo, const float dt)
{
	const int objId = threadIdx.x + blockDim.x * blockIdx.x;
	if (objId >= nObj) return;


	const int3 cid0 = cinfo.getCellIdAlongAxis(props[objId].com);

	const int3 intAxes = make_int3( ceilf(axes+1.0f) );
	const int dx  = threadIdx.y % intAxes.x;
	const int dy  = threadIdx.y / intAxes.x;
	const int dz0 = floorf( axes.z * sqrtf( 1 - sqr(dx*cinfo.h.x / axes.x) - sqr(dy*cinfo.h.y / axes.y) ) );

	for (int dz = dz0-1; dz <= dz0+2; dz++)
	{
		const int cid = cinfo.encode(cid0.x+dx, cid0.y+dy, cid0.z+dz);
		int2 start_size = cellsStartSize[cid];

		for (int pid = start_size.x; pid < start_size.x + start_size.y; pid++)
		{
			float3 coo = make_float3(coosvels[2*pid]);
			float3 vel = make_float3(coosvels[2*pid+1]);

			bounceParticle(coo, vel, dt, props[objId].com, vfts[objId].vel, vfts[objId].omega, axes, vfts[objId].force, vfts[objId].torque);

			// Write back
		}
	}
}
