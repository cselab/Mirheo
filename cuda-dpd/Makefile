override CXXFLAGS += -g -O2 -std=c++11 
NVCCFLAGS = -arch=compute_30 -g -O3 
NVCC = nvcc

test: main.cpp dpd-cuda.o
	$(CXX) $(CXXFLAGS) main.cpp dpd-cuda.o -L/usr/local/cuda-5.5/lib64/ -lcudart -lcurand -o test

dpd-cuda.o: dpd-cuda.cu dpd-cuda.h
	$(NVCC) $(NVCCFLAGS) -c dpd-cuda.cu

clean:
	rm -f test *.o

.PHONY = clean
