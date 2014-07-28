

CXXFLAGS ?= -g -O2 -std=c++11 -I/usr/local/cuda-5.5/include/ -fPIC

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

.DEFAULT_GOAL = test-sem

include ../Makefile

test-sem: main.cpp ../profiler-dpd.o ../cell-lists.o cuda-sem.o
	$(CXX) $(CXXFLAGS) $^ -L/usr/local/cuda-5.5/lib64/ -lcudart -lcurand -o test-sem

libcuda-sem.so: cuda-sem.o ../cell-lists.o ../profiler-dpd.o
	$(CXX) $(CXXFLAGS) -shared $^ -o libcuda-sem.so -L/usr/local/cuda-5.5/lib64/ -lcudart -lcurand

cuda-sem.o: cuda-sem.cu cuda-sem.h
	$(NVCC) $(NVCCFLAGS) -c cuda-sem.cu

../%.o: 
	make -C ../ $(@:../%=%)

clean:
	make -C ../ clean
	rm -f test-sem *.o libcuda-sem.so

.PHONY = clean-sem


