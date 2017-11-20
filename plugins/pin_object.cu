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

__global__ void restrictRigidMotion(ROVview view, int3 pinTranslation, int3 pinRotation, float4* totForces, float4* totTorques)
{
	int objId = blockIdx.x * blockDim.x + threadIdx.x;
	if (objId >= view.nObjects) return;

	auto& motion = view.motions[objId];

	int globObjId = view.ids[objId];

	if (pinTranslation.x) { totForces[globObjId].x  += motion.force.x;   motion.force.x  = 0.0f; }
	if (pinTranslation.y) { totForces[globObjId].y  += motion.force.y;   motion.force.y  = 0.0f; }
	if (pinTranslation.z) { totForces[globObjId].z  += motion.force.z;   motion.force.z  = 0.0f; }

	if (pinRotation.x)    { totTorques[globObjId].x += motion.torque.x;  motion.torque.x = 0.0f; }
	if (pinRotation.y)    { totTorques[globObjId].y += motion.torque.y;  motion.torque.y = 0.0f; }
	if (pinRotation.z)    { totTorques[globObjId].z += motion.torque.z;  motion.torque.z = 0.0f; }
}

__global__ void scaleVec(int n, float4* vectorField, const float a)
{
	const int id = threadIdx.x + blockIdx.x*blockDim.x;
	if (id < n)
		vectorField[id] /= a;
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
	OVview view(ov, ov->local());
	const int nthreads = 128;

	debug("Restricting motionof OV '%s' as per plugin '%s'", ovName.c_str(), name.c_str());

	SAFE_KERNEL_LAUNCH(
			restrictForces,
			view.nObjects, nthreads, 0, stream,
			view, pinTranslation, forces.devPtr() );

	if (rov != nullptr)
	{
		const int nthreads = 32;

		ROVview rovView(rov, rov->local());
		SAFE_KERNEL_LAUNCH(
					restrictRigidMotion,
					getNblocks(view.nObjects, nthreads), nthreads, 0, stream,
					rovView, pinTranslation, pinRotation,
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

