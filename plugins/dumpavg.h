#pragma once
#include "plugin.h"
#include "../core/datatypes.h"

#include <string>

class DumpAvg3D : public Plugin
{
private:
	int sampleEvery;
	int3 resolution;
	bool needDensity, needVelocity, needForce;
	std::string namePrefix;

	PinnedBuffer<float>  density;
	PinnedBuffer<float4> velocity, force;

public:
	DumpAvg3D(int sampleEvery, int3 resolution, bool needDensity, bool needVelocity, bool needForce, std::string namePrefix);
};
