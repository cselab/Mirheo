# since on every mashine it is different 
# For Daint
CUDA_HOME := /opt/nvidia/cudatoolkit/5.5.20-1.0501.7945.8.2/
ARCH_VAL := compute_30
CODE_VAL := sm_30

# For Diego's mashine
#CUDA_HOME := /usr/local/cuda-5.5
#ARCH_VAL := compute_30
#CODE_VAL := sm_30


CXXFLAGS += -g -O2 -std=c++11 -I$(CUDA_HOME)/include -fPIC

#these setting are for my gk104, for k20x pls set the right flags
NVCCFLAGS = -arch=$(ARCH_VAL) -Xcompiler -fpic -code=$(CODE_VAL)  -O4 -g
#NVCCFLAGS += -Xptxas -v,-abi=yes
NVCCFLAGS += --use_fast_math

ifeq "$(profile)" "1"
NVCCFLAGS += -lineinfo -D_PROFILE_ -g
CXXFLAGS += -D_PROFILE_
endif

ifneq "$(debug)" "1"
NVCCFLAGS += -DNDEBUG
else
NVCCFLAGS += -g
endif

CELLING=cell-lists-faster.cu
ifeq "$(legacy)" "1"
CELLING=cell-lists.cu
endif

NVCC = nvcc

test-cell-lists: test-cell-lists.cu cell-lists-faster.cu
	@echo "CUDA_HOME set to " $(CUDA_HOME)
	@echo "CUDA arch is " $(ARCH_VAL)
	$(NVCC) $(NVCCFLAGS) $^ -o test-cell-lists

profiler-dpd.o:	profiler-dpd.cpp profiler-dpd.h
	$(CXX) $(CXXFLAGS) -c profiler-dpd.cpp

cell-lists.o: $(CELLING) cell-lists.h
	$(NVCC) $(NVCCFLAGS) -c $(CELLING) -o $@

clean:
	rm -f test-cell-lists *.o

subclean: clean

.PHONY = clean subclean
