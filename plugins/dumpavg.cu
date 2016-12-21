#include "dumpavg.h"

DumpAvg3D::DumpAvg3D(int sampleEvery, int3 resolution, bool needDensity, bool needVelocity, bool needForce, std::string namePrefix) :
	sampleEvery(sampleEvery), resolution(resolution), density(needDensity), velocity(needVelocity), force(needForce), namePrefix(namePrefix)
{
	const int total = resolution.x * resolution.y * resolution.z;
	if (needDensity)  density .resize(total);
	if (needVelocity) velocity.resize(total);
	if (needForce)    force   .resize(total);
}
