CC=CC
NVCC = nvcc
CXXFLAGS +=  -L$(subst :, -L,$(LD_LIBRARY_PATH))
CXXFLAGS+= -fopenmp -std=c++11 -Wno-unknown-pragmas -Wno-reorder -Wno-comment -Wno-deprecated
GPULIBS+= -lcudart
GPUFLAGS+= -arch=sm_35  -Xcompiler -fpic
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
	$(CC) $^  $(LIBS) -o $@

test-cuda-rbc: main.cudaobj rbc-cuda.cudaobj #rbc.o
	$(CC) $(GPULIBS) $^ -o $@

libcuda-rbc.so: rbcvector-cuda-utils.cudaobj rbc-cuda.cudaobj rbcvector-cpu-utils.o
	$(CC) $(CXXFLAGS) -shared $^ -o $@ -L/usr/local/cuda-6.0/lib64/ -lcudart -lcurand

libcuda-rbc.a: rbcvector-cuda-utils.cudaobj rbc-cuda.cudaobj rbcvector-cpu-utils.o
	ar rcs $@ $^

lib-rbc.so: rbcvector-cpu-utils.o rbc.o
	$(CC) $(CXXFLAGS) -shared -fPIC $^ -o $@

%.o:%.cpp
	$(CC)  $(CXXFLAGS)  $(INCLUDES) -fPIC -c $< -o $@

%.cudaobj:%.cu
	$(NVCC)  $(GPUFLAGS)  $(INCLUDES) -c $< -o $@

%.d: %.cpp
	$(CC)  $(CXXFLAGS) $(THRUST_TARGET_CPU) -MM $< > $*.d

%.dcu: %.cu
	$(NVCC)  $(GPUFLAGS)  $(INCLUDES) -M $< -o $@

