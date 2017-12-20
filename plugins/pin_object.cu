#include "pin_object.h"

#include <core/utils/kernel_launch.h>
#include <core/pvs/object_vector.h>
#include <core/pvs/rigid_object_vector.h>
#include <core/simulation.h>

#include <core/utils/cuda_common.h>
#include <core/utils/cuda_rng.h>
#include "simple_serializer.h"

__global__ void restrictForces(OVview view, int3 pinTranslation, float4* totForces)
{
	int objId = blockIdx.x;

	float3 myf = make_float3(0);

	for (int pid = threadIdx.x; pid < view.objSize; pid += blockDim.x)
	{
		float4 f = view.forces[pid + objId*view.objSize];

		if (pinTranslation.x) { myf.x += f.x; f.x = 0.0f; }
		if (pinTranslation.y) { myf.y += f.y; f.y = 0.0f; }
		if (pinTranslation.z) { myf.z += f.z; f.z = 0.0f; }

		view.forces[pid + objId*view.objSize] = f;
	}

	myf = warpReduce(myf, [] (float a, float b) { return a+b; });

	if (__laneid() == 0)
		atomicAdd(totForces + view.ids[objId], myf);
}

__global__ void revertRigidMotion(ROVviewWithOldMotion view, int3 pinTranslation, int3 pinRotation, float4* totForces, float4* totTorques)
{
	int objId = blockIdx.x * blockDim.x + threadIdx.x;
	if (objId >= view.nObjects) return;

	auto motion     = view.motions    [objId];
	auto old_motion = view.old_motions[objId];

	int globObjId = view.ids[objId];

	if (pinTranslation.x) { totForces[globObjId].x  += old_motion.force.x;   motion.r.x = old_motion.r.x;  motion.vel.x = 0; }
	if (pinTranslation.y) { totForces[globObjId].y  += old_motion.force.y;   motion.r.y = old_motion.r.y;  motion.vel.y = 0; }
	if (pinTranslation.z) { totForces[globObjId].z  += old_motion.force.z;   motion.r.z = old_motion.r.z;  motion.vel.z = 0; }

	// https://stackoverflow.com/a/22401169/3535276
	// looks like q.x, 0, 0, q.w is responsible for the X axis rotation etc.
	// so to restrict rotation along ie. X, we need to preserve q.x
	// and normalize of course
	if (pinRotation.x)    { totTorques[globObjId].x += old_motion.torque.x;  motion.q.y = old_motion.q.y;  motion.omega.x = 0; }
	if (pinRotation.y)    { totTorques[globObjId].y += old_motion.torque.y;  motion.q.z = old_motion.q.z;  motion.omega.y = 0; }
	if (pinRotation.z)    { totTorques[globObjId].z += old_motion.torque.z;  motion.q.w = old_motion.q.w;  motion.omega.z = 0; }

	motion.q = normalize(motion.q);
	view.motions[objId] = motion;
}



void PinObjectPlugin::setup(Simulation* sim, const MPI_Comm& comm, const MPI_Comm& interComm)
{
	SimulationPlugin::setup(sim, comm, interComm);

	ov = sim->getOVbyNameOrDie(ovName);

	int myNObj = ov->local()->nObjects;
	int totObjs;
	MPI_Check( MPI_Allreduce(&myNObj, &totObjs, 1, MPI_INT, MPI_SUM, comm) );

	forces.resize_anew(totObjs);
	forces.clear(0);

	// Also check torques if object is rigid and if we need to restrict rotation
	rov = dynamic_cast<RigidObjectVector*>(ov);
	if (rov != nullptr && (pinRotation.x + pinRotation.y + pinRotation.z) > 0)
	{
		torques.resize_anew(totObjs);
		torques.clear(0);
	}

	debug("Plugin PinObject is setup for OV '%s' and will restrict %s of translational degrees of freedom and %s of rotational",
			ovName.c_str(),

			(std::string(pinTranslation.x ? "x" : "") +
			 std::string(pinTranslation.y ? "y" : "") +
			 std::string(pinTranslation.z ? "z" : "")).c_str(),

			(std::string(pinRotation.x ? "x" : "") +
			 std::string(pinRotation.y ? "y" : "") +
			 std::string(pinRotation.z ? "z" : "")).c_str()   );
}

void PinObjectPlugin::beforeIntegration(cudaStream_t stream)
{
	// If the object is not rigid, modify the forces
	// We'll deal with the rigid objects after the integration
	if (rov == nullptr)
	{
		debug("Restricting motion of OV '%s' as per plugin '%s'", ovName.c_str(), name.c_str());

		const int nthreads = 128;
		OVview view(ov, ov->local());
		SAFE_KERNEL_LAUNCH(
				restrictForces,
				view.nObjects, nthreads, 0, stream,
				view, pinTranslation, forces.devPtr() );
	}
}

void PinObjectPlugin::afterIntegration(cudaStream_t stream)
{
	// If the object IS rigid, revert to old_motion
	if (rov != nullptr)
	{
		debug("Restricting rigid motion of OV '%s' as per plugin '%s'", ovName.c_str(), name.c_str());

		const int nthreads = 32;
		ROVviewWithOldMotion view(rov, rov->local());
		SAFE_KERNEL_LAUNCH(
				revertRigidMotion,
				getNblocks(view.nObjects, nthreads), nthreads, 0, stream,
				view, pinTranslation, pinRotation,
				forces.devPtr(), torques.devPtr() );
	}
}

void PinObjectPlugin::serializeAndSend(cudaStream_t stream)
{
	count++;
	if (count % reportEvery != 0) return;

	forces.downloadFromDevice(stream);
	if (rov != nullptr)
		torques.downloadFromDevice(stream);

	SimpleSerializer::serialize(sendBuffer, ovName, currentTime, reportEvery, forces, torques);\
	send(sendBuffer);

	forces.clearDevice(stream);
	torques.clearDevice(stream);
}


ReportPinObjectPlugin::ReportPinObjectPlugin(std::string name, std::string path) :
				PostprocessPlugin(name), path(path)
{	}

void ReportPinObjectPlugin::setup(const MPI_Comm& comm, const MPI_Comm& interComm)
{
	PostprocessPlugin::setup(comm, interComm);
	activated = createFoldersCollective(comm, path);
}

void ReportPinObjectPlugin::deserialize(MPI_Status& stat)
{
	std::vector<float4> forces, torques;
	float currentTime;
	int nsamples;
	std::string ovName;

	SimpleSerializer::deserialize(data, ovName, currentTime, nsamples, forces, torques);

	MPI_Check( MPI_Reduce( (rank == 0 ? MPI_IN_PLACE : forces.data()),  forces.data(),  forces.size()*4,  MPI_FLOAT, MPI_SUM, 0, comm) );
	MPI_Check( MPI_Reduce( (rank == 0 ? MPI_IN_PLACE : torques.data()), torques.data(), torques.size()*4, MPI_FLOAT, MPI_SUM, 0, comm) );

	std::string tstr = std::to_string(timeStamp++);
	std::string currentFname = path + "/" + ovName + "_" + std::string(5 - tstr.length(), '0') + tstr + ".txt";

	if (activated && rank == 0)
	{
		auto fout = fopen(currentFname.c_str(), "w");

		for (int i=0; i < forces.size(); i++)
		{
			forces[i] /= nsamples;
			fprintf(fout, "%d  %f  %f %f %f", i, currentTime, forces[i].x, forces[i].y, forces[i].z);

			if (i < torques.size())
			{
				torques[i] /= nsamples;
				fprintf(fout, "  %f %f %f", torques[i].x, torques[i].y, torques[i].z);
			}

			fprintf(fout, "\n");
		}

		fclose(fout);
	}
}

