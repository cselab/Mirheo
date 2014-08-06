NVCCFLAGS += -arch=compute_30 -code=sm_30 -O4 -lineinfo

ifneq "$(debug)" "1"
NVCCFLAGS += -DNDEBUG
endif

test:	main.cu
	nvcc $(NVCCFLAGS) main.cu -o test

clean:
	rm test

.PHONY = clean
