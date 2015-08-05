-include .cache.Makefile

CXXFLAGS +=  -L$(subst :, -L,$(LD_LIBRARY_PATH))
CXXFLAGS+= -fopenmp -std=c++11 -Wno-unknown-pragmas -Wno-reorder -Wno-comment -Wno-deprecated
GPULIBS+= -lcudart
GPUFLAGS+= -arch=$(ARCH_VAL)  -code=$(CODE_VAL)  -Xcompiler -fpic
GPUFLAGS+= --use_fast_math

config ?= release

ifeq "$(config)" "debug"
CXXFLAGS+= -O0 -g
GPUFLAGS+= -O0 -g -G -lineinfo
else ifeq "$(config)" "release"
CXXFLAGS+= -O3 -g -fno-tree-vectorize -funroll-loops -DNDEBUG
GPUFLAGS+= -O4 -DNDEBUG
else ifeq "$(config)" "profile"
GPUFLAGS+= -O3 -lineinfo -g
CXXFLAGS+= -O1 -g 
endif

LIBS += #-lstdc++
OS = $(shell uname)
ifneq "$(OS)" "Darwin"
LIBS += -lrt
endif

DIRS = ./
INCLUDES += -I$(DIRS)


.DEFAULT_GOAL := test-cuda-rbc
.PHONY: clean

ifneq "$(MAKECMDGOALS)" "clean"
-include $(notdir $(patsubst %.cpp,%.d,$(wildcard $(DIRS:%/=%/*.cpp))))
endif

ifeq "$(findstring cuda,$(MAKECMDGOALS))" "cuda"
-include $(notdir $(patsubst %.cu,%.dcu,$(wildcard $(DIRS:%/=%/*.cu))))
endif

vpath %.cpp %.h %.cu $(DIRS)

clean:
	rm -f *.cudaobj
	rm -f *.o *.so *.a
	rm -f *.d *.dcu
	rm -f test-rbc
	rm -f test-cuda-rbc

test-rbc: main.o rbcvector-cpu-utils.o rbc.o
	$(CXX) $^  $(LIBS) -o $@

test-cuda-rbc: main.cudaobj rbc-cuda.cudaobj
	$(CXX) $(GPULIBS) $^ -o $@

libcuda-rbc.so: rbc-cuda.cudaobj
	$(CXX) $(CXXFLAGS) -shared $^ -o $@ -L/usr/local/cuda-6.0/lib64/ -lcudart -lcurand

libcuda-rbc.a: rbc-cuda.cudaobj
	ar rcs $@ $^

lib-rbc.so: rbcvector-cpu-utils.o rbc.o
	$(CXX) $(CXXFLAGS) -shared -fPIC $^ -o $@

%.o:%.cpp
	$(CXX)  $(CXXFLAGS)  $(INCLUDES) -fPIC -c $< -o $@

%.cudaobj:%.cu
	$(NVCC)  $(GPUFLAGS)  $(INCLUDES) -c $< -o $@

%.d: %.cpp
	$(CXX)  $(CXXFLAGS) $(THRUST_TARGET_CPU) -MM $< > $*.d

%.dcu: %.cu
	$(NVCC)  $(GPUFLAGS)  $(INCLUDES) -M $< -o $@

