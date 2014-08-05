.SUFFIX: .cu

CXX = g++
CXXFLAGS += -O0 -g -Wall -fmessage-length=0 

all: test_gpu

NVCC = nvcc

test_gpu: rbc_ring.cu
	$(NVCC) -G $(addprefix -Xcompiler ,$(CXXFLAGS)) $< -o $@ -arch=sm_30

test_cpu: rbc_ring.cpp
	$(CXX) -std=c++11 $(CXXFLAGS) $< -o $@

clean:
	rm -f test_cpu test_gpu
