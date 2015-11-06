-include ../../mpi-dpd/.cache.Makefile

NVCC ?= nvcc -ccbin $(CXX)
ARCH_VAL ?= compute_35
CODE_VAL ?= sm_35


NVCCFLAGS += -I$(HDF5_DIR)/include -Xcudafe "--diag_suppress=unrecognized_gcc_pragma"
NVCCFLAGS +=  -arch $(ARCH_VAL) -code $(CODE_VAL) -O3 -use_fast_math -g -DNDEBUG -Xcompiler "-fopenmp"
CXXFLAGS +=  -L../../cuda-dpd/dpd -L../../cuda-rbc/ -L../../cuda-ctc/ -O3 -g  -std=c++11 -DNDEBUG -fopenmp
NVCCFLAGS += -I../../cuda-dpd/dpd -I../../cuda-dpd/ -I../../cuda-rbc/ -I../../cuda-ctc -I../../mpi-dpd
LIBS = -lcuda-dpd -lcuda-rbc -lcuda-ctc -lcudart -ldl -lz  -fopenmp

testcontact: ../../mpi-dpd/test testcontact.o
	make -C ../../mpi-dpd
	rm -f ../../mpi-dpd/main.o
	$(CXX) $(CXXFLAGS) testcontact.o ../../mpi-dpd/*.o $(LIBS) -o testcontact

../test:
	make -C ../
	cp ../*.o .

testcontact.o: testcontact.cu ../../mpi-dpd/argument-parser.h ../../mpi-dpd/common.h ../../mpi-dpd/containers.h ../../mpi-dpd/contact.h ../../cuda-dpd/dpd-rng.h
		$(NVCC) $(NVCCFLAGS) testcontact.cu -c -o testcontact.o

.PHONY: ../test
