include ../Makefile

ifeq "$(nonperiodic)" "1"
    NVCCFLAGS += -D_NONPERIODIC_KERNEL_
endif

.DEFAULT_GOAL = test-dpd

test-dpd: main.cpp libcuda-dpd.so
	$(CXX) $(CXXFLAGS) $^  -lcudart -lcurand -o test-dpd

libcuda-dpd.so: cuda-dpd.o cuda-dpd-bipartite.o ../profiler-dpd.o celllists
	$(CXX) $(CXXFLAGS) -shared cuda-dpd.o cuda-dpd-bipartite.o ../profiler-dpd.o ../cell-lists.o -o libcuda-dpd.so  -lcudart -lcurand

cuda-dpd.o: cuda-dpd.cu cuda-dpd.h
	$(NVCC) $(NVCCFLAGS) -c cuda-dpd.cu

cuda-dpd-bipartite.o: cuda-dpd-bipartite.cu cuda-dpd.h
	$(NVCC) $(NVCCFLAGS) -c cuda-dpd-bipartite.cu

../%.o:
	make -C ../ $(@:../%=%)

celllists:
	make -C ../ cell-lists.o NVCCFLAGS="$(NVCCFLAGS)"

clean:
	rm -f test-dpd libcuda-dpd.so *.o
	make -C ../ clean

subclean: clean

.PHONY = clean subclean celllists
