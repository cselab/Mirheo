#pragma once

#include <mpi.h>
#include <plugins/interface.h>
#include <memory>

class Postprocess
{
private:
	MPI_Comm comm;
	MPI_Comm interComm;
	std::vector< std::unique_ptr<PostprocessPlugin> > plugins;
	std::vector<MPI_Request> requests;

public:
	Postprocess(MPI_Comm& comm, MPI_Comm& interComm);
	void registerPlugin( std::unique_ptr<PostprocessPlugin> plugin );
	void run();
};
