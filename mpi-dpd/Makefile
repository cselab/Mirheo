NVCC=nvcc

NVCCFLAGS += -g -I/opt/openmpi/include -arch sm_30 -O3 -use_fast_math -lineinfo -DNDEBUG
NVCCFLAGS += -D_WITHCUDA_ -I../cuda-dpd-sem/dpd -L/opt/openmpi/lib -L../cuda-dpd-sem/dpd/

OBJS = main.o dpd-interactions.o redistribute-particles.o common.o 
#../cuda-dpd-sem/dpd/libcuda-dpd.so

test: $(OBJS) libcuda-dpd
	echo HELLO
	$(NVCC) $(NVCCFLAGS) $(OBJS) -lcuda-dpd -lmpi_cxx -lmpi -lcudart -o test

%.o: %.cu %.h common.h
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

main.o: main.cu common.h
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

libcuda-dpd: 
	make -C ../cuda-dpd-sem/dpd libcuda-dpd.so nonperiodic=1

clean:
	rm -f test *.o

#we want to give always a try to make a new libcuda-dpd - too complicated to track dependecies there
.PHONY = clean libcuda-dpd
