MPI_CC=CC
#MPI_CC=mpic++
CPPFLAGS = -g -O2

DEFINES= -DPACKAGE_NAME=\"OSU-Micro-Benchmarks\" -DPACKAGE_TARNAME=\"osu-micro-benchmarks\" -DPACKAGE_VERSION=\"4.4.1\" -DPACKAGE_STRING=\"OSU-Micro-Benchmarks\ 4.4.1\"  \
-DPACKAGE_BUGREPORT=\"mvapich-discuss@cse.ohio-state.edu\" -DPACKAGE=\"osu-micro-benchmarks\" -DVERSION=\"4.4.1\" -DSTDC_HEADERS=1 -DHAVE_SYS_TYPES_H=1 -DHAVE_SYS_STAT_H=1 -DHAVE_STDLIB_H=1  \
-DHAVE_STRING_H=1 -DHAVE_MEMORY_H=1 -DHAVE_STRINGS_H=1 -DHAVE_INTTYPES_H=1 -DHAVE_STDINT_H=1 -DHAVE_UNISTD_H=1 -DHAVE_STDLIB_H=1 -DHAVE_STRING_H=1 -DHAVE_SYS_TIME_H=1 -DHAVE_UNISTD_H=1  \
-DHAVE_GETPAGESIZE=1 -DHAVE_GETTIMEOFDAY=1 -DHAVE_MEMSET=1 -DHAVE_SQRT=1 -DFIELD_WIDTH=18 -DFLOAT_PRECISION=2 

all: halo_bench_isend halo_bench_persist halo_bench_send osu_latency osu_latency_rdp mesh_topo mesh_distances byte_latency

halo_bench_isend: halo_bench.cpp
	$(MPI_CC) $(CPPFLAGS) -DUSE_MPI_ISEND=1 -DUSE_MPI_IRECV=1 -o halo_bench_isend halo_bench.cpp

halo_bench_send: halo_bench.cpp
	$(MPI_CC) $(CPPFLAGS) -DUSE_MPI_SEND=1 -DUSE_MPI_IRECV=1 -o halo_bench_send halo_bench.cpp

halo_bench_persist: halo_bench.cpp
	$(MPI_CC) $(CPPFLAGS) -DUSE_MPI_PERS_SEND=1 -DUSE_MPI_PERS_RECV=1 -o halo_bench_persist halo_bench.cpp

osu_latency: osu_latency.c
	$(MPI_CC) $(CFLAGS) $(DEFINES) -o osu_latency osu_latency.c

osu_latency_rdp: osu_latency_rdp.cpp
	$(MPI_CC) $(CFLAGS) -o osu_latency_rdp osu_latency_rdp.cpp

mesh_topo: mesh_topo.cpp
	$(MPI_CC) $(CFLAGS) -o mesh_topo mesh_topo.cpp

mesh_distances: mesh_distances.cpp
	$(MPI_CC) $(CFLAGS) -o mesh_distances mesh_distances.cpp

byte_latency: byte_latency.cpp
	$(MPI_CC) $(CFLAGS) -o byte_latency byte_latency.cpp

clean:
	rm -f halo_bench_isend halo_bench_send halo_bench_persist osu_latency osu_latency_rdp mesh_topo mesh_distances byte_latency
