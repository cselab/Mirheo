#include <core/logger.h>
#include <core/task_scheduler.h>

#define private public

Logger logger;

int main(int argc, char ** argv)
{
	MPI_Init(&argc, &argv);
	logger.init(MPI_COMM_WORLD, "scheduler.log", 9);

	//  A1,A2 - B -----------
	//              \        \
	//                D1,D2 - E
	//          C - /
	//              \ F
	//                        G

	TaskScheduler scheduler;

	scheduler.addTask("C", [](){ printf("c\n"); });
	scheduler.addTask("G", [](){ printf("g\n"); });
	scheduler.addTask("D", [](){ printf("d1\n"); });
	scheduler.addTask("A", [](){ printf("a1\n"); });
	scheduler.addTask("E", [](){ printf("e\n"); });
	scheduler.addTask("A", [](){ printf("a2\n"); });
	scheduler.addTask("B", [](){ printf("b\n"); });
	scheduler.addTask("D", [](){ printf("d1\n"); });
	scheduler.addTask("F", [](){ printf("f\n"); });

	scheduler.addDependency("B", {"A"});
	scheduler.addDependency("D", {"B", "C"});
	scheduler.addDependency("F", {"C"});
	scheduler.addDependency("E", {"D"});
	scheduler.addDependency("E", {"B"});

	scheduler.compile();
	scheduler.run();

	return 0;
}
