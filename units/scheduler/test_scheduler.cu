#include <core/logger.h>
#include <core/task_scheduler.h>

#include "timer.h"

#define private public

Logger logger;

int main(int argc, char ** argv)
{
	int provided;
	MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
	if (provided < MPI_THREAD_MULTIPLE)
	{
		printf("ERROR: The MPI library does not have full thread support\n");
		MPI_Abort(MPI_COMM_WORLD, 1);
	}
	MPI_Errhandler_set(MPI_COMM_WORLD, MPI_ERRORS_RETURN);

	logger.init(MPI_COMM_WORLD, "scheduler.log", 9);

	//  A1,A2 - B -----------
	//              \        \
	//                D1,D2 - E
	//          C - /
	//              \ F
	//                        G

	{
		TaskScheduler scheduler;

		scheduler.addTask("C", [](cudaStream_t s){ printf("c\n"); });
		scheduler.addTask("G", [](cudaStream_t s){ printf("g\n"); });
		scheduler.addTask("D", [](cudaStream_t s){ printf("d2\n"); });
		scheduler.addTask("A", [](cudaStream_t s){ printf("a1\n"); });
		scheduler.addTask("E", [](cudaStream_t s){ printf("e\n"); });
		scheduler.addTask("A", [](cudaStream_t s){ printf("a2\n"); });
		scheduler.addTask("B", [](cudaStream_t s){ printf("b\n"); });
		scheduler.addTask("D", [](cudaStream_t s){ printf("d1\n"); });
		scheduler.addTask("F", [](cudaStream_t s){ printf("f\n"); });

		scheduler.addDependency("B", {}, {"A"});
		scheduler.addDependency("D", {}, {"B", "C"});
		scheduler.addDependency("F", {}, {"C"});
		scheduler.addDependency("E", {}, {"D"});
		scheduler.addDependency("E", {}, {"B"});

		scheduler.compile();
		scheduler.run();
	}

	printf("Benchmarking\n");

	TaskScheduler scheduler;

	float a, b, c, d, e, f, g;
	a=b=c=d=e=f=g = 0;
	scheduler.addTask("C", [&](cudaStream_t s){ c++; });
	scheduler.addTask("G", [&](cudaStream_t s){ g--; });
	scheduler.addTask("D", [&](cudaStream_t s){ d+=2; });
	scheduler.addTask("A", [&](cudaStream_t s){ a-=3; });
	scheduler.addTask("E", [&](cudaStream_t s){ e*=1.001; });
	scheduler.addTask("A", [&](cudaStream_t s){ a*=0.9999; });
	scheduler.addTask("B", [&](cudaStream_t s){ b+=5; });
	scheduler.addTask("D", [&](cudaStream_t s){ d-=42; });
	scheduler.addTask("F", [&](cudaStream_t s){ f*=2; });

	scheduler.addDependency("B", {}, {"A"});
	scheduler.addDependency("D", {}, {"B", "C"});
	scheduler.addDependency("F", {}, {"C"});
	scheduler.addDependency("E", {}, {"D"});
	scheduler.addDependency("E", {}, {"B"});

	scheduler.compile();

	Timer timer;
	timer.start();

	int n = 10000;
	for (int i=0; i<n; i++)
		scheduler.run();

	int64_t tm = timer.elapsed();

	printf("Per run: %f us\n", (double)tm / (1000.0*n));

	return 0;
}
