NVCC = nvcc
NVCCFLAGS = -std=c++11 -arch=compute_61 -code=sm_61 -O3 -DNDEBUG --use_fast_math -g -lineinfo --expt-extended-lambda

LIBS = -lmpi

onerank_obj = ../core/scan.o ../core/celllist.o ../core/halo_exchanger.o ../core/dpd.o test_onerank.o
onerank_cu  = $(patsubst %.o,%.cu,$(onerank_obj))
onerank_dep = $(patsubst %.o,%.d, $(onerank_obj))

halo_obj = ../core/scan.o ../core/celllist.o ../core/halo_exchanger.o test_halo.o
halo_cu  = $(patsubst %.o,%.cu,$(halo_obj))
halo_dep = $(patsubst %.o,%.d, $(halo_obj))

log_obj = test_logger.o
log_cu  = $(patsubst %.o,%.cu,$(log_obj))
log_dep = $(patsubst %.o,%.d, $(log_obj))


ifeq "$(MAKECMDGOALS)" "test_onerank"
-include $(onerank_dep)
endif

ifeq "$(MAKECMDGOALS)" "test_halo"
-include $(halo_dep)
endif

ifeq "$(MAKECMDGOALS)" "test_logger"
-include $(log_dep)
endif

default:
	@echo "Please specify a target!"

test_logger: $(log_obj)
	$(NVCC) $(NVCCFLAGS) $(LIBS) $^ -o $@

test_onerank: $(onerank_obj)
	$(NVCC) $(NVCCFLAGS) $(LIBS) $^ -o $@
	
test_halo: $(halo_obj)
	$(NVCC) $(NVCCFLAGS) $(LIBS) $^ -o $@

%.d: %.cu
	$(NVCC) $(NVCCFLAGS) -M $< > $@
	
%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@
	
clean:
	rm -f $(onerank_obj) $(halo_obj) $(onerank_dep) $(halo_dep) test_onerank test_halo
	
.PHONY: clean default