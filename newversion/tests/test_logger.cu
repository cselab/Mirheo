// Yo ho ho ho
#define private public

#include "../core/logger.h"

int main(int argc, char ** argv)
{
	int rank;
	MPI_Init(&argc, &argv);
	Logger logger(MPI_COMM_WORLD, "dummy.log");

	logger.say("hello hello hello lalala");
	logger.say("this shouldn't be overlapping!    ");
	logger.warn("This is warning");
	logger.say("11111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111");
	logger.debug("debug invisible");
	logger.MPI_Check(MPI_Comm_rank(MPI_COMM_WORLD, &rank));

	logger.debugLvl() = 9;
	logger.debug("debug here!");
	logger.die("oh, this was an error");
}
