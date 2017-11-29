#pragma once

#include <mpi.h>
#include <plugins/interface.h>

class Postprocess
{
private:
	MPI_Comm comm;
	MPI_Comm interComm;
	std::vector<PostprocessPlugin*> plugins;
	std::vector<MPI_Request> requests;

public:
	Postprocess(MPI_Comm& comm, MPI_Comm& interComm);
	void registerPlugin(PostprocessPlugin* plugin);
	void run();
};
