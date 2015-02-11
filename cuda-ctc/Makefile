CXX ?= g++
NVCC ?= nvcc
CXXFLAGS +=  -L$(subst :, -L,$(LD_LIBRARY_PATH))
CXXFLAGS+= -fopenmp -std=c++11 -Wno-unknown-pragmas -Wno-reorder -Wno-comment -Wno-deprecated
GPULIBS+= -lcuda -lcudart -L/usr/local/cuda/lib64
GPUFLAGS+= -arch=compute_30  -Xcompiler -fpic -code=sm_30
GPUFLAGS+= --use_fast_math

config ?= release

ifeq "$(config)" "debug"
debug=1
CXXFLAGS+= -O0 -g
GPUFLAGS+= -O0 -g -lineinfo
else ifeq "$(config)" "release"
debug=0
CXXFLAGS+= -O3 -g -fno-tree-vectorize -funroll-loops -DNDEBUG
GPUFLAGS+= -O4 -DNDEBUG
else ifeq "$(config)" "profile"
debug=0
GPUFLAGS+= -O3 -lineinfo -g
CXXFLAGS+= -O1 -g 
endif

LIBS += #-lstdc++
OS = $(shell uname)
ifneq "$(OS)" "Darwin"
LIBS += -lrt
endif

DIRS = ./
INCLUDES += -I$(DIRS) -I../cuda-dpd-sem/dpd/
LIBCUDADPD = -L../cuda-dpd-sem/dpd/ -lcuda-dpd
vpath %.cpp %.h %.cu $(DIRS)

.DEFAULT_GOAL := test-cuda-ctc
.PHONY: clean $(LIBCUDADPD)

ifneq "$(MAKECMDGOALS)" "clean"
-include $(notdir $(patsubst %.cpp,%.d,$(wildcard $(DIRS:%/=%/*.cpp))))
endif

ifeq "$(findstring cuda,$(MAKECMDGOALS))" "cuda"
-include $(notdir $(patsubst %.cu,%.dcu,$(wildcard $(DIRS:%/=%/*.cu))))
endif


clean:
	rm -f *.cudaobj
	rm -f *.o
	rm -f *.d
	rm -f test-rbc
	rm -f test-cuda-rbc


test-cuda-ctc: main.cudaobj ctc-cuda.cudaobj $(LIBCUDADPD)
	$(CXX) $(GPULIBS) $^ -o $@

libcuda-ctc.so: ctc-cuda.cudaobj
	$(CXX) $(CXXFLAGS) -shared $^ -o $@ -L/usr/local/cuda/lib64/ -lcudart -lcurand

$(LIBCUDADPD):
	make -C ../cuda-dpd-sem/dpd libcuda-dpd.so nonperiodic=1 debug=$(debug)

%.o: %.cpp
	$(CXX)  $(CXXFLAGS)  $(INCLUDES) -fPIC -c $< -o $@

%.cudaobj: %.cu
	$(NVCC)  $(GPUFLAGS)  $(INCLUDES) -c $< -o $@

%.d: %.cpp
	$(CXX)  $(CXXFLAGS) $(INCLUDES) -MM $< > $*.d

%.dcu: %.cu
	$(NVCC)  $(GPUFLAGS)  $(INCLUDES) -M $< -o $@
	sed -i "s/$(basename $<).o/$(basename $<).cudaobj/" $@

