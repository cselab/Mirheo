#pragma once

#include <core/utils/cuda_common.h>
#include <core/walls/sdf_wall.h>

template<typename T>
__device__ __forceinline__ float evalSdf(T x, SDFWall::SdfInfo sdfInfo)
{
	float3 x3{x.x, x.y, x.z};
	float3 texcoord = floorf((x3 + sdfInfo.extendedDomainSize*0.5f) * sdfInfo.invh);
	float3 lambda = (x3 - (texcoord * sdfInfo.h - sdfInfo.extendedDomainSize*0.5f)) * sdfInfo.invh;

	const float s000 = tex3D<float>(sdfInfo.sdfTex, texcoord.x + 0, texcoord.y + 0, texcoord.z + 0);
	const float s001 = tex3D<float>(sdfInfo.sdfTex, texcoord.x + 1, texcoord.y + 0, texcoord.z + 0);
	const float s010 = tex3D<float>(sdfInfo.sdfTex, texcoord.x + 0, texcoord.y + 1, texcoord.z + 0);
	const float s011 = tex3D<float>(sdfInfo.sdfTex, texcoord.x + 1, texcoord.y + 1, texcoord.z + 0);
	const float s100 = tex3D<float>(sdfInfo.sdfTex, texcoord.x + 0, texcoord.y + 0, texcoord.z + 1);
	const float s101 = tex3D<float>(sdfInfo.sdfTex, texcoord.x + 1, texcoord.y + 0, texcoord.z + 1);
	const float s110 = tex3D<float>(sdfInfo.sdfTex, texcoord.x + 0, texcoord.y + 1, texcoord.z + 1);
	const float s111 = tex3D<float>(sdfInfo.sdfTex, texcoord.x + 1, texcoord.y + 1, texcoord.z + 1);

	const float s00x = s000 * (1 - lambda.x) + lambda.x * s001;
	const float s01x = s010 * (1 - lambda.x) + lambda.x * s011;
	const float s10x = s100 * (1 - lambda.x) + lambda.x * s101;
	const float s11x = s110 * (1 - lambda.x) + lambda.x * s111;

	const float s0yx = s00x * (1 - lambda.y) + lambda.y * s01x;
	const float s1yx = s10x * (1 - lambda.y) + lambda.y * s11x;

	const float szyx = s0yx * (1 - lambda.z) + lambda.z * s1yx;

	return szyx;
}
