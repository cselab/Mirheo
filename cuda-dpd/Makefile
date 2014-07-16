override CXXFLAGS += -g -O2 -std=c++11 
NVCCFLAGS = -arch=compute_30 -g -O4 -DNDEBUG
#-Xcompiler -fpic
NVCC = nvcc

test: main.cpp cuda-dpd.o rring-buffer.o 
	$(CXX) $(CXXFLAGS) main.cpp cuda-dpd.o rring-buffer.o -L/usr/local/cuda-5.5/lib64/ -lcudart -lcurand -o test

cuda-dpd.so: dpd-cuda.o rring-buffer.o
	$(CXX) $(CXXFLAGS) -shared cuda-dpd.o -o cuda-dpd.so -L/usr/local/cuda-5.5/lib64/ -lcudart -lcurand

cuda-dpd.o: cuda-dpd.cu cuda-dpd.h rring-buffer.o
	$(NVCC) $(NVCCFLAGS) -c cuda-dpd.cu

rring-buffer.o: rring-buffer.cu rring-buffer.h
	$(NVCC) $(NVCCFLAGS) -c rring-buffer.cu

clean:
	rm -f test *.o

.PHONY = clean
