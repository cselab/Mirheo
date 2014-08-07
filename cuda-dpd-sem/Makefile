CXXFLAGS += -g -O2 -std=c++11 -I/opt/nvidia/cudatoolkit/5.5.20-1.0501.7945.8.2/include -fPIC

#these setting are for my gk104, for k20x pls set the right flags
NVCCFLAGS = -arch=compute_30  -Xcompiler -fpic -code=sm_30  -O4
#NVCCFLAGS += -Xptxas -v,-abi=yes
NVCCFLAGS += -Xptxas -dlcm=cg
NVCCFLAGS += --maxrregcount 30
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

NVCC = nvcc

profiler-dpd.o:	profiler-dpd.cpp profiler-dpd.h
	$(CXX) $(CXXFLAGS) -c profiler-dpd.cpp

cell-lists.o: cell-lists.cu cell-lists.h
	$(NVCC) $(NVCCFLAGS) -c cell-lists.cu

clean:
	rm -f test *.o

subclean: clean

.PHONY = clean subclean
