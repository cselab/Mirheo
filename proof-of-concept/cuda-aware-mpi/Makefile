-include .cache.Makefile


NVCCFLAGS += -g -O2 -arch sm_30 -L/opt/openmpi/lib -I/opt/openmpi/include

test: main.o
	$(CXX) main.o -lcudart -o test

main.o: main.cu
	$(NVCC) $(NVCCFLAGS) main.cu -c

clean:
	rm -f test *.o

.PHONY = clean
