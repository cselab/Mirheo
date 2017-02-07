// Yo ho ho ho
#define private public

#include <core/logger.h>

Logger logger;

int main(int argc, char ** argv)
{
	int rank;
	MPI_Init(&argc, &argv);
	logger.init(MPI_COMM_WORLD, "dummy.log");

	say("hello hello hello lalala");
	say("this shouldn't be overlapping!    ");
	warn("This is warning");
	say("11111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111");
	debug("debug invisible");
	MPI_Check(MPI_Comm_rank(MPI_COMM_WORLD, &rank));

	logger.debugLvl() = 9;
	debug("debug here!");
	die("oh, this was an error");
}
