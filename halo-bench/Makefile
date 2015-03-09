MPI_CC=CC
#MPI_CC=mpic++

all: halo_bench_isend halo_bench_send halo_bench_persist

halo_bench_isend: halo_bench.cpp
	$(MPI_CC) -g -O2 -DUSE_MPI_ISEND=1 -DUSE_MPI_IRECV=1 -o halo_bench_isend halo_bench.cpp

halo_bench_send: halo_bench.cpp
	$(MPI_CC) -g -O2 -DUSE_MPI_SEND=1 -DUSE_MPI_IRECV=1 -o halo_bench_send halo_bench.cpp

halo_bench_persist: halo_bench.cpp
	$(MPI_CC) -g -O2 -DUSE_MPI_PERS_SEND=1 -DUSE_MPI_PERS_RECV=1 -o halo_bench_persist halo_bench.cpp

clean:
	rm -f halo_bench_isend halo_bench_send halo_bench_persist
