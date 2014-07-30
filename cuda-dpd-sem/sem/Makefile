include ../Makefile

.DEFAULT_GOAL = test-sem

compilemain := $(CXX) $(CXXFLAGS) -c main.cpp
ifeq "$(gpu-only)" "1"
compilemain := $(NVCC) $(NVCCFLAGS) -c main.cu
endif

test-sem: main.o ../profiler-dpd.o ../cell-lists.o cuda-sem.o
	$(CXX) $(CXXFLAGS) $^ -L/usr/local/cuda-5.5/lib64/ -lcudart -lcurand -o test-sem

libcuda-sem.so: cuda-sem.o ../cell-lists.o ../profiler-dpd.o
	$(CXX) $(CXXFLAGS) -shared $^ -o libcuda-sem.so -L/usr/local/cuda-5.5/lib64/ -lcudart -lcurand

main.o: main.cu main.cpp
	$(compilemain) -c $(mainsrc)

cuda-sem.o: cuda-sem.cu cuda-sem.h
	$(NVCC) $(NVCCFLAGS) -c cuda-sem.cu

../%.o: 
	make -C ../ $(@:../%=%)

clean:
	make -C ../ clean
	rm -f test-sem *.o libcuda-sem.so

.PHONY = clean-sem main.o


