override CXXFLAGS += -g -O2 -std=c++11 
NVCCFLAGS = -arch=compute_30 -g -Xcompiler -fpic -O4
#-O4 -DNDEBUG

NVCC = nvcc

test: main.cpp cuda-dpd.o rring-buffer.o profiler-dpd.o
	$(CXX) $(CXXFLAGS) main.cpp cuda-dpd.o rring-buffer.o profiler-dpd.o -L/usr/local/cuda-5.5/lib64/ -lcudart -lcurand -o test

libcuda-dpd.so: cuda-dpd.o rring-buffer.o profiler-dpd.o
	$(CXX) $(CXXFLAGS) -shared $^ -o libcuda-dpd.so -L/usr/local/cuda-5.5/lib64/ -lcudart -lcurand

cuda-dpd.o: cuda-dpd.cu cuda-dpd.h
	$(NVCC) $(NVCCFLAGS) -c cuda-dpd.cu

rring-buffer.o: rring-buffer.cu rring-buffer.h
	$(NVCC) $(NVCCFLAGS) -c rring-buffer.cu

profiler-dpd.o:	profiler-dpd.cu profiler-dpd.h
	$(NVCC) $(NVCCFLAGS) -c profiler-dpd.cu

clean:
	rm -f test libcuda-dpd.so *.o 

.PHONY = clean
