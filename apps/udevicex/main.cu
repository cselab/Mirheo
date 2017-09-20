#include <core/argument_parser.h>

#include <core/udevicex.h>
#include <core/parser.h>

Logger logger;

int main(int argc, char** argv)
{
	int rank;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	// Get the script filename
	std::string xmlname;
	std::vector<ArgumentParser::OptionStruct> opts
	({
		{'i', "input",  ArgumentParser::STRING, "Input script", &xmlname, std::string("script.xml")},
	});

	ArgumentParser::Parser parser(opts, rank == 0);
	parser.parse(argc, argv);

	// Parse the script
	Parser udxParser(xmlname);
	uDeviceX* udevicex = udxParser.setup_uDeviceX(logger);

	// Shoot
	udevicex->run(udxParser.getNIterations());

	return 0;
}
