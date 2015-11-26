#include <mpi.h>

#include <cstdio>

#define MPI_CHECK(ans) do { mpiAssert((ans), __FILE__, __LINE__); } while(0)

inline void mpiAssert(int code, const char *file, int line, bool abort=true)
{
    if (code != MPI_SUCCESS)
    {
	char error_string[2048];
	int length_of_error_string = sizeof(error_string);
	MPI_Error_string(code, error_string, &length_of_error_string);

	printf("mpiAssert: %s %d %s\n", file, line, error_string);

	MPI_Abort(MPI_COMM_WORLD, code);
    }
}
