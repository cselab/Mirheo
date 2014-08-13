include ../Makefile

.DEFAULT_GOAL = test-sem

compilemain := $(CXX) $(CXXFLAGS) -c main.cpp
ifeq "$(gpu-only)" "1"
compilemain := $(NVCC) $(NVCCFLAGS) -c main.cu
endif

test-sem: main.o libcuda-sem.so
	$(CXX) $(CXXFLAGS) $^ -L$(CUDA_HOME)/lib64/ -lcudart -lcurand -o test-sem

libcuda-sem.so: cuda-sem.o  ../profiler-dpd.o  cell-factory.o #../cell-lists.o
	@echo "CUDA_HOME is" $(CUDA_HOME)
	$(CXX) $(CXXFLAGS) -shared $^ -o libcuda-sem.so -L$(CUDA_HOME)/lib64/ -lcudart -lcurand

main.o: main.cu main.cpp
	$(compilemain) -c $(mainsrc)

cuda-sem.o: cuda-sem.cu cuda-sem.h
	$(NVCC) $(NVCCFLAGS) -c cuda-sem.cu

cell-factory.o: cell-factory.cu cell-factory.h
	$(NVCC) $(NVCCFLAGS) -c cell-factory.cu

../%.o: 
	make -C ../ $(@:../%=%)

clean:
	make -C ../ clean
	rm -f test-sem *.o libcuda-sem.so

.PHONY = clean-sem main.o


