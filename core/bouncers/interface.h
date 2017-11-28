#pragma once

#include <string>

class CellList;
class ParticleVector;
class ObjectVector;

/**
 * Interface for a class implementing bouncing from objects
 */
class Bouncer
{
protected:
	ObjectVector* ov;  /// Particles will be bounced against that ObjectVector

	/**
	 * Should be defined to implement bounce
	 *
	 * \param pv ptr to ParticleVector whose particles will be
	 * bounced from the objects associated with this bouncer
	 * \param cl ptr to CellList that has to be built for pv
	 * \param dt timestep used to integrate pv
	 * \param local if true, will bounce from the local objects, if false -- from halo objects
	 * Note that particles from pv (that actually will be bounced back) are always local
	 * \param stream cuda stream on which to execute
	 */
	virtual void exec (ParticleVector* pv, CellList* cl, float dt, bool local, cudaStream_t stream) = 0;

public:
	std::string name;

	Bouncer(std::string name) : name(name) {};

	/**
	 * Second step of initialization, called from the Simulation
	 * All the preparation for bouncing must be made here
	 */
	virtual void setup(ObjectVector* ov) = 0;

	/**
	 * Ask ParticleVectors which the class will be working with to have specific properties
	 * Default: ask nothing
	 * Called from Simulation right after setup
	 */
	virtual void setPrerequisites(ParticleVector* pv) {}

	/// Interface to the private exec function for local objects
	void bounceLocal(ParticleVector* pv, CellList* cl, float dt, cudaStream_t stream) { exec (pv, cl, dt, true,  stream); }

	/// Interface to the private exec function for halo objects
	void bounceHalo (ParticleVector* pv, CellList* cl, float dt, cudaStream_t stream) { exec (pv, cl, dt, false, stream); }

	virtual ~Bouncer() = default;
};
