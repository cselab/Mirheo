include ../Makefile

.DEFAULT_GOAL = test-dpd

test-dpd: main.cpp libcuda-dpd.so #cuda-dpd.o cuda-dpd-bipartite.o ../profiler-dpd.o ../cell-lists.o
	$(CXX) $(CXXFLAGS) $^ -L/usr/local/cuda-5.5/lib64/ -lcudart -lcurand -o test-dpd

libcuda-dpd.so: cuda-dpd.o cuda-dpd-bipartite.o ../cell-lists.o ../profiler-dpd.o
	$(CXX) $(CXXFLAGS) -shared $^ -o libcuda-dpd.so -L/usr/local/cuda-5.5/lib64/ -lcudart -lcurand

cuda-dpd.o: cuda-dpd.cu cuda-dpd.h
	$(NVCC) $(NVCCFLAGS)  --maxrregcount 30  -Xptxas -dlcm=cg -c cuda-dpd.cu

cuda-dpd-bipartite.o: cuda-dpd-bipartite.cu cuda-dpd.h
	$(NVCC) $(NVCCFLAGS)  --maxrregcount 30  -Xptxas -dlcm=cg -c cuda-dpd-bipartite.cu

../%.o:
	make -C ../ $(@:../%=%)

clean:
	rm -f test-dpd libcuda-dpd.so *.o
	make -C ../ clean

subclean: clean 

.PHONY = clean subclean
