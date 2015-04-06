-include .cache.Makefile

include ../Makefile

CUDADPD = cuda-dpd.cu
CUDADPDBIP = cuda-dpd-bipartite.cu

ifeq "$(floatized)" "1"
CUDADPD = cuda-dpd-floatized.cu
CUDADPDBIP = cuda-dpd-bipartite-floatized.cu
endif

ifeq "$(nonperiodic)" "1"
    NVCCFLAGS += -D_NONPERIODIC_KERNEL_
endif

ifeq "$(MAKECMDGOALS)" "libcuda-dpd.so"
CXXFLAGS += -fPIC
NVCCFLAGS += -Xcompiler -fpic
endif

NVCCFLAGS += -DVISCOSITY_S_LEVEL=$(slevel) -lineinfo -Xptxas -v

.DEFAULT_GOAL = test-dpd

test-dpd: main.cpp libcuda-dpd.so
	$(CXX) $(CXXFLAGS) $^  -lcudart -lcurand -o test-dpd

libcuda-dpd.so: cuda-dpd.o cuda-dpd-bipartite.o ../profiler-dpd.o celllists
	$(CXX) $(CXXFLAGS) -shared cuda-dpd.o cuda-dpd-bipartite.o ../profiler-dpd.o ../cell-lists.o -o libcuda-dpd.so  -lcudart -lcurand

libcuda-dpd.a: cuda-dpd.o cuda-dpd-bipartite.o ../profiler-dpd.o celllists
	ar rcs $@ cuda-dpd.o cuda-dpd-bipartite.o ../profiler-dpd.o ../cell-lists.o

cuda-dpd.o: $(CUDADPD) cuda-dpd.h
	$(NVCC) $(NVCCFLAGS) -cubin $(CUDADPD)
	$(NVCC) $(NVCCFLAGS) -c $(CUDADPD) -o $@

cuda-dpd-bipartite.o: $(CUDADPDBIP) cuda-dpd.h
	$(NVCC) $(NVCCFLAGS) -c $(CUDADPDBIP) -o $@

../%.o:
	make -C ../ $(@:../%=%) CXX="$(CXX)" NVCC="$(NVCC)"

celllists:
	make -C ../ cell-lists.o NVCC="$(NVCC)" NVCCFLAGS="$(NVCCFLAGS)"

clean:
	rm -f test-dpd libcuda-dpd.so *.o *.a
	make -C ../ clean

subclean: clean

.PHONY = clean subclean celllists
