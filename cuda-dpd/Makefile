override CXXFLAGS += -g -O2 -std=c++11 -I/usr/local/cuda-5.5/include/

NVCCFLAGS = -arch=compute_30 -Xptxas -dlcm=cg -O4 -Xcompiler -fpic

ifeq "$(profile)" "1"
NVCCFLAGS += -lineinfo -D_PROFILE_ -g
endif

ifneq "$(debug)" "1"
NVCCFLAGS += -DNDEBUG
else
NVCCFLAGS += -g
endif

NVCC = nvcc

test: main.cpp cuda-dpd.o rring-buffer.o profiler-dpd.o
	$(CXX) $(CXXFLAGS) main.cpp cuda-dpd.o rring-buffer.o profiler-dpd.o -L/usr/local/cuda-5.5/lib64/ -lcudart -lcurand -o test

libcuda-dpd.so: cuda-dpd.o rring-buffer.o profiler-dpd.o
	$(CXX) $(CXXFLAGS) -shared $^ -o libcuda-dpd.so -L/usr/local/cuda-5.5/lib64/ -lcudart -lcurand

cuda-dpd.o: cuda-dpd.cu cuda-dpd.h
	$(NVCC) $(NVCCFLAGS) -c cuda-dpd.cu

rring-buffer.o: rring-buffer.cpp rring-buffer.h
	$(CXX) $(CXXFLAGS) -c rring-buffer.cpp

profiler-dpd.o:	profiler-dpd.cpp profiler-dpd.h
	$(CXX) $(CXXFLAGS) -c profiler-dpd.cpp

clean:
	rm -f test libcuda-dpd.so *.o 

.PHONY = clean
