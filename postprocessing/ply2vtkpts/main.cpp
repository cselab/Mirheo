#include <omp.h>
#include <mpi.h>

#include <cstdio>
#include <cassert>
#include <vector>
#include <algorithm>


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

using namespace std;

#include <vtkVersion.h>
#include <vtkCellArray.h>
#include <vtkPoints.h>
#include <vtkXMLPolyDataWriter.h>
#include <vtkPolyData.h>
#include <vtkSmartPointer.h>

int dump_vtk_points(const char * dstpath, const int nrbcs, const float * xs, const float * ys, const float * zs )
{
    vtkSmartPointer<vtkPoints> points =
	vtkSmartPointer<vtkPoints>::New();

    for ( unsigned int i = 0; i < nrbcs; ++i )
	points->InsertNextPoint ( xs[i], ys[i], zs[i] );

    // Create a polydata object and add the points to it.
    vtkSmartPointer<vtkPolyData> polydata =
	vtkSmartPointer<vtkPolyData>::New();
    polydata->SetPoints(points);

    // Write the file
    vtkSmartPointer<vtkXMLPolyDataWriter> writer =
	vtkSmartPointer<vtkXMLPolyDataWriter>::New();
    writer->SetFileName(dstpath);
#if VTK_MAJOR_VERSION <= 5
    writer->SetInput(polydata);
#else
    writer->SetInputData(polydata);
#endif

    writer->Write();

    return EXIT_SUCCESS;
}

int main(int argc, char ** argv)
{
    MPI_CHECK(MPI_Init(&argc, &argv));

    int nranks, rank;
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &nranks));
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));

    const bool verbose = false;

    if (argc != 4)
    {
	if (rank == 0)
	    printf("usage: test <number-of-points-per-cell> <path-to-ply-file> <dst-path>\n");

	exit(EXIT_FAILURE);
    }

    const int nvpc = atoi(argv[1]);
    const char * path = argv[2];
    const char * dstpath = argv[3];

    if (rank == 0)
	printf("reading at location <%s>\n", path);

    int nallvertices, nallrbcs, headersize;

    const double tstart = omp_get_wtime();

    if (rank == 0)
    {
	FILE * f = fopen(path, "r");
	assert(f);
	char line[2048];

	auto eat_line = [&] ()
	    {
		fgets(line, 2048, f);

		if (verbose)
		    printf("reading <%s>\n", line);
	    };

	for(int i = 0; i < 3; ++i)
	    eat_line();

	int retval = sscanf(line, "element vertex %d\n", &nallvertices);
	assert(retval == 1);

        nallrbcs = nallvertices / nvpc;

	if (verbose)
	    printf("*** nvertices: %d\n", nallvertices);

	for(int i = 0; i < 7; ++i)
	    eat_line();

	int nfaces = -1;
	retval = sscanf(line, "element face %d\n", &nfaces);
	assert(retval == 1);

	if (verbose)
	    printf("*** nfaces: %d\n", nfaces);

	for(int i = 0; i < 2; ++i)
	    eat_line();

	headersize = ftell(f);

	fclose(f);
    }

    MPI_CHECK(MPI_Bcast(&nallvertices, 1, MPI_INT, 0, MPI_COMM_WORLD));
    MPI_CHECK(MPI_Bcast(&nallrbcs, 1, MPI_INT, 0, MPI_COMM_WORLD));
    MPI_CHECK(MPI_Bcast(&headersize, 1, MPI_INT, 0, MPI_COMM_WORLD));

    const double theader = omp_get_wtime();

    const int myrbcs_size = nallrbcs / nranks + (int)(rank < (nallrbcs % nranks));
    const int myrbcs_start = nallrbcs / nranks * rank + min(rank, nallrbcs % nranks);

    float * data = new float[6 * myrbcs_size * nvpc];

    {
	MPI_File filehandle;
	MPI_CHECK( MPI_File_open(MPI_COMM_WORLD, path, MPI_MODE_RDONLY, MPI_INFO_NULL, &filehandle) );

	MPI_Status status;
	MPI_CHECK( MPI_File_read_at(filehandle, headersize + myrbcs_start * nvpc * 6 * sizeof(float),
				    data, myrbcs_size * nvpc * 6, MPI_FLOAT, &status));

	MPI_CHECK( MPI_File_close(&filehandle));
    }

    vector<float> coords[3];

    for(int i = 0; i < 3; ++i)
	coords[i].resize(myrbcs_size);

#pragma omp parallel for
    for(int r = 0; r < myrbcs_size; ++r)
    {
	float com[3] = {0, 0, 0};

        for(int v = 0;  v < nvpc; ++v)
	    for(int c = 0; c < 3; ++c)
		com[c] += data[c + 6 * (v + nvpc * r)];

	for(int i = 0; i < 3; ++i)
	    coords[i][r] = com[i] /nvpc;
    }

    delete [] data;

    vector<float> allcoords[3];

    if (rank == 0)
	for(int i = 0; i < 3; ++i)
	    allcoords[i].resize(nranks * ((nallrbcs + nranks - 1) / nranks));

    for(int i = 0; i < 3; ++i)
    {
	MPI_CHECK( MPI_Gather(&coords[i].front(), nallrbcs / nranks, MPI_FLOAT,
			      &allcoords[i].front(), nallrbcs / nranks, MPI_FLOAT,
			      0, MPI_COMM_WORLD) );

	MPI_CHECK( MPI_Gather(&coords[i].back(), 1, MPI_FLOAT,
			      (&allcoords[i].front()) + nranks * (nallrbcs / nranks), 1, MPI_FLOAT,
			      0, MPI_COMM_WORLD) );
    }

    const double tthroughput = omp_get_wtime();

    if (rank == 0)
	dump_vtk_points(dstpath, nallrbcs, &allcoords[0].front(), &allcoords[1].front(), &allcoords[2].front());

    const double tvtk = omp_get_wtime();

    if (rank == 0)
    {
	const double ttotal = tvtk - tstart;

	printf("TOTAL TIME: %.2f\n", ttotal);

	printf("TDISTRIBUTION: HEADER:%.1f%%\tI/O+REDUCE:%.1f%%\tVTK:%.1f%%\t\n",
	       100 / ttotal * (theader - tstart),
	       100 / ttotal * (tthroughput - theader),
	       100 / ttotal * (tvtk - tthroughput));

	const double memfp_ply = 6. * nallvertices * sizeof(float) / pow(1024., 3);
	const double memfp_vtk = 3. * nallrbcs * sizeof(float) / pow(1024., 3);

	printf("THROUGHPUT: %.1f GB/s\n", (memfp_ply + memfp_vtk) / (tthroughput - theader));
	printf("VTK DUMP: %.1f GB/s\n", memfp_vtk / (tvtk - tthroughput));
    }

    MPI_CHECK(MPI_Finalize());

    return 0;
}

