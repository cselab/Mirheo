// Yo ho ho ho
#define private public
#define protected public

#include <core/utils/cuda_common.h>
#include <core/containers.h>
#include <core/pvs/particle_vector.h>
#include <core/pvs/rigid_object_vector.h>
#include <core/celllist.h>
#include <core/logger.h>
#include <core/mesh.h>
#include <core/bouncers/interface.h>
#include <core/bouncers/from_mesh.h>
#include <core/initial_conditions/rigid_ic.h>


#include <unistd.h>

#include <fstream>
#include <string>

Logger logger;


int main(int argc, char ** argv)
{
	// Init

	int provided;
	MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
	if (provided < MPI_THREAD_MULTIPLE)
	{
	    printf("ERROR: The MPI library does not have full thread support\n");
	    MPI_Abort(MPI_COMM_WORLD, 1);
	}

	logger.init(MPI_COMM_WORLD, "leak.log", 9);
	srand48(time(0));

	float3 length{12, 12, 12};

	Mesh mesh("../../walls/sphere.off");
	float radius = 5.0;

	ObjectVector* obj = new RigidObjectVector("obj", 1, {1, 1, 1}, 2, std::move(mesh));
	ParticleVector* pv = new ParticleVector("pv", 1);
	CellList *cl = new PrimaryCellList(pv, 1.0f, length);

	pv->domain = obj->domain = {length, make_float3(0), length};

	RigidIC *ic = new RigidIC("obj.xyz", "ic.ic");
	ic->exec(MPI_COMM_WORLD, obj, obj->domain, 0);


	printf("initializing...\n");
	int np = 25000000;
	pv->requireDataPerParticle<Particle>("old_particles", false);
	pv->local()->resize_anew(np);
	std::vector<Particle> initial(np);


	float h = 0.2;
	float dt = 0.1;

	auto coosvels = pv->local()->coosvels.hostPtr();
	auto old = pv->local()->extraPerParticle.getData<Particle>("old_particles")->hostPtr();

	for (int i=0; i<np; i++)
	{
		float r = radius + h*drand48();

		float u = drand48();
		float v = drand48();

		float theta = 2*M_PI*u;
		float phi = acos(2*v - 1);

		float3 x = make_float3(r*cos(theta)*sin(phi), r*sin(theta)*sin(phi), r*cos(phi));

		Particle p, pOld;
		p.i1 = pOld.i1 = i;
		p.r = x;
		p.u = x/dt;
		pOld.r = make_float3(0);
		pOld.u = make_float3(0);

		coosvels[i] = p;
		old[i] = pOld;
	}

	printf("generated %d particles\n", np);
	pv->local()->coosvels.uploadToDevice(0);

	cl->build(0);

	pv->local()->coosvels.downloadFromDevice(0);
	for (int i=0; i<np; i++)
		initial[i] = pv->local()->coosvels[i];

	Bouncer* bouncer = new BounceFromMesh("bounce", 0.1);
	bouncer->setup(obj);

	bouncer->bounceLocal(pv, cl, dt, 0);

	CUDA_Check( cudaDeviceSynchronize() );

	pv->local()->coosvels.downloadFromDevice(0);
	coosvels = pv->local()->coosvels.hostPtr();

	for (int i=0; i<np; i++)
	{
		float rho = sqrt(dot(coosvels[i].r, coosvels[i].r));

		if (rho > radius)
			printf("%d failed (rho %f): [%f %f %f] to [%f %f %f]\n",
					i, rho,
					initial[i].r.x,  initial[i].r.y,  initial[i].r.z,
					coosvels[i].r.x, coosvels[i].r.y, coosvels[i].r.z);
	}


	return 0;
}
