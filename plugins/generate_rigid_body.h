#pragma once

#include <plugins/plugin.h>

#include <string>
#include <fstream>

class GenRigid : public SimulationPlugin
{
private:
	std::string pvName, fname;
	int genTimeStep;
	float3  x0, axes;

public:
	GenRigid(std::string name, std::string pvName, std::string fname, int genTimeStep, float3 x0, float3 axes) :
		SimulationPlugin(name), pvName(pvName), fname(fname), genTime(genTime), x0(x0), axes(axes)
	{ }

	void afterIntegration()
	{
		if (currentTimeStep != genTimeStep)
			return;

		auto pv = sim->getPVbyName(name);
		if (pv == nullptr)
			die("No such PV registered: %s", name.c_str());

		auto sqr = [] (float x) {
			return x*x;
		};

		const float3 invaxes = 1.0f / axes;
		auto ellipsoid = [invaxes, x0] (const float3 r) {
			const float3 tmp = (r-x0) * invaxes;
			return dot(tmp, tmp);
		};

		pv->local()->coosvels.downloadFromDevice();
		auto ptr = pv->local()->coosvels.hostPtr();
		std::vector<float3> res;

		for (int i=0; i<pv->local()->size(); i++)
			if (ellipsoid(ptr[i].r) <= 1.0f)
				res.push_back(ptr[i].r - x0);

		std::ofstream fout(fname);
		if (!fout.good()) die("Couldn't open file %s", fname.c_str());

		fout << res.size() << std::endl;
		fout << "# " << axes.x << "  " << axes.y << "  " << axes.z << std::endl;
		for (auto r : res)
			fout << r.x << " " << r.y << " " << r.z << std::endl;

		fout.close();
	}

	~GenRigid() {};
};
