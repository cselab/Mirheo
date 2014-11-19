-include .cache.Makefile

NVCC ?= nvcc

ARCH_VAL ?= compute_35
CODE_VAL ?= sm_35

NVCCFLAGS += -g -I/opt/openmpi/include -arch $(ARCH_VAL) -code $(CODE_VAL) -g -O3 -use_fast_math -lineinfo -DNDEBUG
NVCCFLAGS += -I../cuda-dpd-sem/dpd -L/opt/openmpi/lib -L../cuda-dpd-sem/dpd/

OBJS = main.o dpd-interactions.o redistribute-particles.o halo-exchanger.o common.o wall-interactions.o
LIBS=  -lcuda-dpd -lcudart

ifneq (,$(findstring openmpi,$(shell which $(CXX))))
	LIBS +=  -lmpi_cxx -lmpi
else
	LIBS += -lmpichcxx -lmpich -lcudart
endif

test: $(OBJS) libcuda-dpd
	echo HELLO
	$(NVCC) $(NVCCFLAGS) $(OBJS) $(LIBS) -o test

%.o: %.cu %.h common.h
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

main.o: main.cu common.h
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

libcuda-dpd: 
	make -C ../cuda-dpd-sem/dpd libcuda-dpd.so nonperiodic=1

clean:
	rm -f test *.o

cleanall: clean
	make -C ../cuda-dpd-sem/dpd clean

#we want to give always a try to make a new libcuda-dpd - too complicated to track dependecies there
.PHONY = clean libcuda-dpd
