#pragma once

#include <string>
#include <core/datatypes.h>
#include <core/containers.h>
#include <core/domain.h>

#include "extra_data/extra_data_manager.h"

#include <map>

class ParticleVector;

class LocalParticleVector
{
public:
	ParticleVector* pv;

	PinnedBuffer<Particle> coosvels;
	DeviceBuffer<Force> forces;
	ExtraDataManager extraPerParticle;

	// Local coordinate system; (0,0,0) is center of the local domain
	LocalParticleVector(ParticleVector* pv, int n=0) : pv(pv)
	{
		resize_anew(n);
	}

	int size()
	{
		return np;
	}

	virtual void resize(const int n, cudaStream_t stream)
	{
		if (n < 0) die("Tried to resize PV to %d < 0 particles", n);

		//debug4("Resizing PV '%s' (%s) from %d particles to %d particles",
		//		pv->name.c_str(), this == pv->local() ? "local" : "halo",
		//		np, n);

		coosvels.        resize(n, stream);
		forces.          resize(n, stream);
		extraPerParticle.resize(n, stream);

		np = n;
	}

	virtual void resize_anew(const int n)
	{
		if (n < 0) die("Tried to resize PV to %d < 0 particles", n);

		coosvels.        resize_anew(n);
		forces.          resize_anew(n);
		extraPerParticle.resize_anew(n);

		np = n;
	}

	virtual ~LocalParticleVector() = default;

protected:
	int np;
};


class ParticleVector
{
public:
	DomainInfo domain;
	LocalParticleVector *_local, *_halo;

	float mass;
	std::string name;
	// Local coordinate system; (0,0,0) is center of the local domain

	bool haloValid = false;
	bool redistValid = false;

	int cellListStamp{0};

	ParticleVector(std::string name, float mass, int n=0) :
		ParticleVector(
				name, mass,
				new LocalParticleVector(this, n),
				new LocalParticleVector(this, 0) )
	{	}

	LocalParticleVector* local() { return _local; }
	LocalParticleVector* halo()  { return _halo;  }

	virtual void checkpoint(MPI_Comm comm, std::string path);
	virtual void restart(MPI_Comm comm, std::string path);

	template<typename T>
	void requireDataPerParticle(std::string name, bool needExchange)
	{
		requireDataPerParticle<T>(name, needExchange, 0);
	}

	template<typename T>
	void requireDataPerParticle(std::string name, bool needExchange, int shiftDataType)
	{
		requireDataPerParticle<T>(local(), name, needExchange, shiftDataType);
		requireDataPerParticle<T>(halo(),  name, needExchange, shiftDataType);
	}

	virtual ~ParticleVector() { delete _local; delete _halo; }

protected:
	ParticleVector(	std::string name, float mass, LocalParticleVector *local, LocalParticleVector *halo ) :
		name(name), mass(mass), _local(local), _halo(halo)
	{
		// usually old positions and velocities don't need to exchanged
		requireDataPerParticle<Particle> ("old_particles", false);
	}

	int restartIdx = 0;

private:

	template<typename T>
	void requireDataPerParticle(LocalParticleVector* lpv, std::string name, bool needExchange, int shiftDataType)
	{
		lpv->extraPerParticle.createData<T> (name, lpv->size());
		if (needExchange) lpv->extraPerParticle.requireExchange(name);
		if (shiftDataType != 0) lpv->extraPerParticle.requireShift(name, shiftDataType);
	}
};

#include "views/pv.h"



