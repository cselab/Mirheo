ARCH_VAL ?= compute_35
CODE_VAL ?= sm_35

CXXFLAGS += -g -O2  -fPIC

#these setting are for my gk104, for k20x pls set the right flags
NVCCFLAGS = -arch=$(ARCH_VAL) -Xcompiler -fpic -code=$(CODE_VAL)  -O4
#-DNDEBUG
#NVCCFLAGS += -Xptxas -v,-abi=yes
NVCCFLAGS += --use_fast_math
# -Xptxas -dlcm=cg -maxrregcount 32

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
