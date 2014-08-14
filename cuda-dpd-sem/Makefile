CXXFLAGS += -g -O2 -std=c++11 -I/usr/local/cuda-5.5/include -fPIC

#these setting are for my gk104, for k20x pls set the right flags
NVCCFLAGS = -arch=compute_30  -Xcompiler -fpic -code=sm_30  -O4 -g
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
	$(NVCC) $(NVCCFLAGS) $^ -o test-cell-lists

profiler-dpd.o:	profiler-dpd.cpp profiler-dpd.h
	$(CXX) $(CXXFLAGS) -c profiler-dpd.cpp

cell-lists.o: $(CELLING) cell-lists.h
	$(NVCC) $(NVCCFLAGS) -c $(CELLING) -o $@

clean:
	rm -f test-cell-lists *.o

subclean: clean

.PHONY = clean subclean
