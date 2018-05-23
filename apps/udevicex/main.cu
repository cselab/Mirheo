#include <core/argument_parser.h>

#include <core/udevicex.h>
#include <core/parser/parser.h>

Logger logger;

int main(int argc, char** argv)
{
	int rank;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	// Get the script filename
	std::string xmlname;
	bool gpuAwareMPI;
	std::vector<ArgumentParser::OptionStruct> opts
	({
		{'i', "input",          ArgumentParser::STRING, "Input script",                            &xmlname,     std::string("script.xml")},
		{'g', "gpu-aware-mpi",  ArgumentParser::BOOL,   "Use GPU/CUDA aware MPI (e.g. GPUDirect)", &gpuAwareMPI, false},
	});

	ArgumentParser::Parser parser(opts, rank == 0);
	parser.parse(argc, argv);

	{
		// Parse the script
		Parser udxParser(xmlname);
		auto udevicex = std::move( udxParser.setup_uDeviceX(logger, gpuAwareMPI) );

		// Shoot
		udevicex->run(udxParser.getNIterations());
	}

	MPI_Finalize();

	return 0;
}
