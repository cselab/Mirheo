NVCC=nvcc

NVCCFLAGS += -g -O2 -arch sm_30 -L/opt/openmpi/lib -I/opt/openmpi/include

test: main.cu
	$(NVCC) $(NVCCFLAGS) main.cu -lmpi_cxx -lmpi -lcudart -o test

clean:
	rm test

.PHONY = clean
