#include <gtest/gtest.h>
#include <string>
#include <vector>

#include <core/logger.h>
#include <core/task_scheduler.h>

#include "../timer.h"

#define private public

Logger logger;

TEST(Scheduler, Order)
{
    //  A1,A2 - B -----------
    //              \        \
    //                D1,D2 - E
    //          C - /
    //              \ F
    //                        G

    TaskScheduler scheduler;
    std::vector<std::string> messages;

    auto A1 = scheduler.createTask("A1");
    auto A2 = scheduler.createTask("A2");
    auto B  = scheduler.createTask("B");
    auto C  = scheduler.createTask("C");
    auto D1 = scheduler.createTask("D1");
    auto D2 = scheduler.createTask("D2");
    auto E  = scheduler.createTask("E");
    auto F  = scheduler.createTask("F");
    auto G  = scheduler.createTask("G");

    scheduler.addTask(A1, [&](cudaStream_t s){ messages.push_back("a1"); });
    scheduler.addTask(A2, [&](cudaStream_t s){ messages.push_back("a2"); });
    scheduler.addTask(B , [&](cudaStream_t s){ messages.push_back("b" ); });
    scheduler.addTask(C , [&](cudaStream_t s){ messages.push_back("c" ); });
    scheduler.addTask(D1, [&](cudaStream_t s){ messages.push_back("d1"); });
    scheduler.addTask(D2, [&](cudaStream_t s){ messages.push_back("d2"); });
    scheduler.addTask(E , [&](cudaStream_t s){ messages.push_back("e" ); });
    scheduler.addTask(F , [&](cudaStream_t s){ messages.push_back("f" ); });
    scheduler.addTask(G , [&](cudaStream_t s){ messages.push_back("g" ); });
                
    scheduler.addDependency(B, {}, {A1, A2});
    scheduler.addDependency(D1, {}, {B, C});
    scheduler.addDependency(D2, {}, {B, C});
    scheduler.addDependency(F, {}, {C});
    scheduler.addDependency(E, {}, {D1, D2, B});

    scheduler.compile();
    scheduler.run();

    ASSERT_EQ(messages.size(), 9);

    // TODO
}

TEST(Scheduler, Benchmark)
{
    TaskScheduler scheduler;

    float a, b, c, d, e, f, g;
    a = b = c = d = e = f = g = 0;
        
    auto A1 = scheduler.createTask("A1");
    auto A2 = scheduler.createTask("A2");
    auto B  = scheduler.createTask("B");
    auto C  = scheduler.createTask("C");
    auto D1 = scheduler.createTask("D1");
    auto D2 = scheduler.createTask("D2");
    auto E  = scheduler.createTask("E");
    auto F  = scheduler.createTask("F");
    auto G  = scheduler.createTask("G");
        
    scheduler.addTask(C,  [&](cudaStream_t s){ c++; });
    scheduler.addTask(G,  [&](cudaStream_t s){ g--; });
    scheduler.addTask(D1, [&](cudaStream_t s){ d+=2; });
    scheduler.addTask(A1, [&](cudaStream_t s){ a-=3; });
    scheduler.addTask(E,  [&](cudaStream_t s){ e*=1.001; });
    scheduler.addTask(A2, [&](cudaStream_t s){ a*=0.9999; });
    scheduler.addTask(B,  [&](cudaStream_t s){ b+=5; });
    scheduler.addTask(D2, [&](cudaStream_t s){ d-=42; });
    scheduler.addTask(F,  [&](cudaStream_t s){ f*=2; });

    scheduler.addDependency(B, {}, {A1, A2});
    scheduler.addDependency(D1, {}, {B, C});
    scheduler.addDependency(D2, {}, {B, C});
    scheduler.addDependency(F, {}, {C});
    scheduler.addDependency(E, {}, {D1, D2, B});

    scheduler.compile();

    Timer timer;
    timer.start();

    int n = 10000;
    for (int i=0; i<n; i++)
        scheduler.run();

    int64_t tm = timer.elapsed();

    double tus = (double)tm / (1000.0*n);
    fprintf(stderr, "Per run: %f us\n", tus);

    EXPECT_LE(tus, 500.0);
}

int main(int argc, char **argv)
{
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    if (provided < MPI_THREAD_MULTIPLE) {
        fprintf(stderr, "ERROR: The MPI library does not have full thread support\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    MPI_Errhandler_set(MPI_COMM_WORLD, MPI_ERRORS_RETURN);
    logger.init(MPI_COMM_WORLD, "scheduler.log", 9);

    testing::InitGoogleTest(&argc, argv);
    
    auto ret = RUN_ALL_TESTS();

    MPI_Finalize();
    return ret;
}
