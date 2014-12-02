-include .cache.Makefile

NVCC ?= nvcc
ARCH_VAL ?= compute_35
CODE_VAL ?= sm_35

NVCCFLAGS +=  -arch $(ARCH_VAL) -code $(CODE_VAL) -O3 -use_fast_math -lineinfo -g
#-DNDEBUG
CXXFLAGS += -I../cuda-dpd-sem/dpd  -L../cuda-dpd-sem/dpd  -L../cuda-rbc/ -I../cuda-rbc/ -g -O3
#-DNDEBUG
NVCCFLAGS += -I../cuda-dpd-sem/dpd  -L../cuda-dpd-sem/dpd  -I../cuda-rbc/ -L../cuda-rbc/

OBJS = main.o dpd-interactions.o redistribute-particles.o halo-exchanger.o common.o wall-interactions.o scan.o redistribute-rbcs.o rbc-interactions.o
LIBS = -lcuda-dpd -lcuda-rbc -lcudart 

ifneq "$(h5part)" "0"
LIBS += -lH5Part
endif

test: $(OBJS) libcuda-dpd libcuda-rbc
	echo HELLO
	$(CXX) $(CXXFLAGS) $(OBJS) $(LIBS) -o test

%.o: %.cu %.h common.h
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

main.o: main.cu common.h
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

libcuda-dpd: 
	make -C ../cuda-dpd-sem/dpd libcuda-dpd.so nonperiodic=1  CXX="$(CXX)" NVCC="$(NVCC)" ARCH_VAL="$(ARCH_VAL)" CODE_VAL="$(CODE_VAL)"

libcuda-rbc: 
	make -C ../cuda-rbc/ libcuda-rbc.so  CXX="$(CXX)" NVCC="$(NVCC)" ARCH_VAL="$(ARCH_VAL)" CODE_VAL="$(CODE_VAL)"

clean:
	rm -f test *.o

cleanall: clean
	make -C ../cuda-dpd-sem/dpd clean
	make -C ../cuda-rbc clean

#we want to give always a try to make a new libcuda-dpd - too complicated to track dependecies there
.PHONY = clean libcuda-dpd libcuda-rbc
