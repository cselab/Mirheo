include ../Makefile

.DEFAULT_GOAL = test-dpd

test-dpd: main.cpp cuda-dpd.o libcuda-dpd.so
	$(CXX) $(CXXFLAGS) $^ -L$(CUDA_HOME)/lib64/ -lcudart -lcurand -o test-dpd

libcuda-dpd.so: cuda-dpd.o cuda-dpd-bipartite.o ../cell-lists.o ../profiler-dpd.o
	@echo "CUDA_HOME set to " $(CUDA_HOME)
	$(CXX) $(CXXFLAGS) -shared $^ -o libcuda-dpd.so -L$(CUDA_HOME)/lib64/ -lcudart -lcurand

cuda-dpd.o: cuda-dpd.cu cuda-dpd.h
	$(NVCC) $(NVCCFLAGS) -c cuda-dpd.cu

cuda-dpd-bipartite.o: cuda-dpd-bipartite.cu cuda-dpd.h
	$(NVCC) $(NVCCFLAGS) -c cuda-dpd-bipartite.cu

../%.o:
	make -C ../ $(@:../%=%)

clean:
	rm -f test-dpd libcuda-dpd.so *.o
	make -C ../ clean

subclean: clean

.PHONY = clean subclean
