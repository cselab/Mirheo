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
	std::string variables;
	int forceDebugLvl;
	std::vector<ArgumentParser::OptionStruct> opts
	({
		{'i', "input",          ArgumentParser::STRING, "Input script",                             &xmlname,     std::string("script.xml")},
		{'g', "gpu-aware-mpi",  ArgumentParser::BOOL,   "Use GPU/CUDA aware MPI (e.g. GPUDirect)",  &gpuAwareMPI, false},
		{'d', "debug",          ArgumentParser::INT,    "Override debug level, "
				                                        "from 1 (lowest) to 9 (highest)",           &forceDebugLvl, -1},

		{'v', "variables",      ArgumentParser::STRING, "Comma-separated list of variables and "
				                                        "their values that will be substituted "
                                                        "into the input script with format: "
				                                        "'var1=value1,var2=value2, var3=value3'",   &variables,     std::string("")},
	});

	ArgumentParser::Parser parser(opts, rank == 0);
	parser.parse(argc, argv);

	{
		// Parse the script
		Parser udxParser(xmlname, forceDebugLvl, variables);
		auto udevicex = std::move( udxParser.setup_uDeviceX(logger, gpuAwareMPI) );

		// Shoot
		udevicex->run(udxParser.getNIterations());
	}

	MPI_Finalize();

	return 0;
}
