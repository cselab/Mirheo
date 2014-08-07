include ../Makefile

.DEFAULT_GOAL = test-dpd

test-dpd: main.cpp cuda-dpd.o cuda-dpd-bipartite.o ../profiler-dpd.o ../cell-lists.o
	$(CXX) $(CXXFLAGS) $^ -L/opt/nvidia/cudatoolkit/5.5.20-1.0501.7945.8.2/lib64 -lcudart -lcurand -o test-dpd

libcuda-dpd.so: cuda-dpd.o cuda-dpd-bipartite.o ../cell-lists.o ../profiler-dpd.o
	$(CXX) $(CXXFLAGS) -shared $^ -o libcuda-dpd.so -L/opt/nvidia/cudatoolkit/5.5.20-1.0501.7945.8.2/lib64 -lcudart -lcurand

cuda-dpd.o: cuda-dpd.cu cuda-dpd.h
	$(NVCC) $(NVCCFLAGS) -c cuda-dpd.cu

cuda-dpd-bipartite.o: cuda-dpd-bipartite.cu cuda-dpd.h
	$(NVCC) $(NVCCFLAGS) -c cuda-dpd-bipartite.cu

../%.o:
	make -C ../ $(@:../%=%)

clean:
	rm -f test-dpd libcuda-dpd.so *.o

subclean: clean

.PHONY = clean subclean
