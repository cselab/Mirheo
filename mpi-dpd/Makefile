-include .cache.Makefile

NVCC ?= nvcc
ARCH_VAL ?= compute_35
CODE_VAL ?= sm_35

NVCCFLAGS += -I$(HDF5_DIR)/include -Xcudafe "--diag_suppress=unrecognized_gcc_pragma"
NVCCFLAGS +=  -arch $(ARCH_VAL) -code $(CODE_VAL) -O3 -use_fast_math -g  -DNDEBUG 
CXXFLAGS += -L../cuda-dpd-sem/dpd -L../cuda-rbc/ -L../cuda-ctc/ -O3 -g  
#-DNDEBUG
NVCCFLAGS += -I../cuda-dpd-sem/dpd -I../cuda-rbc/ -I../cuda-ctc

OBJS = 	dpd-interactions.o \
	redistribute-particles.o \
	halo-exchanger.o common.o \
	wall-interactions.o scan.o \
	redistribute-rbcs.o \
	rbc-interactions.o \
	containers.o \
	io.o \
	main.o 

LIBS = -lcuda-dpd -lcuda-rbc -lcuda-ctc -lcudart 

ifneq "$(h5part)" "0"
LIBS += -L$(HDF5_DIR)/lib -lhdf5
else
NVCCFLAGS += -DNO_H5PART
endif

ifneq "$(h5)" "0"
LIBS += -lhdf5
else
NVCCFLAGS += -DNO_H5
endif

slevel ?= 1

NVCCFLAGS += -DVISCOSITY_S_LEVEL=$(slevel)

inquire: $(shell [ `cat slevel.txt` == "$(slevel)" ] || { echo "cleanall" ; } )
	$(shell echo "$(slevel)" > slevel.txt )
	$(MAKE) $(MFLAGS) test

test:  $(OBJS) libcuda-dpd libcuda-rbc libcuda-ctc
	$(CXX) $(CXXFLAGS) $(OBJS) $(LIBS) -o test

-include $(OBJS:.o=.d)

%.o: %.cu
	$(NVCC) $(NVCCFLAGS) $*.cu -c -o $*.o 
	$(NVCC) $(NVCCFLAGS) -M $*.cu > $*.d

libcuda-dpd: 
	make -C ../cuda-dpd-sem/dpd libcuda-dpd.so nonperiodic=1 CXX="$(CXX)" NVCC="$(NVCC)" ARCH_VAL="$(ARCH_VAL)" CODE_VAL="$(CODE_VAL)" slevel="$(slevel)"

libcuda-rbc: 
	make -C ../cuda-rbc/ libcuda-rbc.so  CXX="$(CXX)" NVCC="$(NVCC)" ARCH_VAL="$(ARCH_VAL)" CODE_VAL="$(CODE_VAL)"

libcuda-ctc: 
	make -C ../cuda-ctc/ libcuda-ctc.so  CXX="$(CXX)" NVCC="$(NVCC)" ARCH_VAL="$(ARCH_VAL)" CODE_VAL="$(CODE_VAL)"

clean:
	rm -f test *.o *.d

cleanall: clean
	rm -f slevel.txt
	make -C ../cuda-dpd-sem/dpd clean
	make -C ../cuda-rbc clean
	make -C ../cuda-ctc clean

#we want to give always a try to update the libs - too complicated to track dependecies there
.PHONY = clean libcuda-dpd libcuda-rbc libcuda-ctc inquire
