NVCC = nvcc
NVCCFLAGS = -std=c++11 -arch=compute_61 -code=sm_61 -O3 -DNDEBUG --use_fast_math -g -lineinfo --expt-extended-lambda

LIBS = -lmpi

SRC_DIR = ../../core/ ../
VPATH = $(SRC_DIR)
NVCCFLAGS += $(addprefix -I,$(SRC_DIR))

onerank_obj = scan.o celllist.o halo_exchanger.o dpd.o test_onerank.o
halo_obj    = scan.o celllist.o halo_exchanger.o test_halo.o
log_obj     = test_logger.o
cells_obj   = scan.o celllist.o test_cells.o
dpd_obj     = scan.o celllist.o dpd.o test_dpd.o


default:
	@echo "Please specify a target!"

test_cells: $(cells_obj)
	$(NVCC) $(NVCCFLAGS) $(LIBS) $^ -o $@

test_logger: $(log_obj)
	$(NVCC) $(NVCCFLAGS) $(LIBS) $^ -o $@

test_onerank: $(onerank_obj)
	$(NVCC) $(NVCCFLAGS) $(LIBS) $^ -o $@
	
test_halo: $(halo_obj)
	$(NVCC) $(NVCCFLAGS) $(LIBS) $^ -o $@
	
test_dpd: $(dpd_obj)
	$(NVCC) $(NVCCFLAGS) $(LIBS) $^ -o $@

%.d: %.cu
	@$(NVCC) $(NVCCFLAGS) -M $< > $(patsubst %.d,%.TMPd,$@)
	@mv $(patsubst %.d,%.TMPd,$@) $@
	
%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@
	
clean:
	rm -f *.o *.d test_onerank test_halo test_logger test_cells test_dpd
	
.PHONY: clean default


ifneq "$(MAKECMDGOALS)" "clean"
-include $(patsubst %.cu,%.d,$(notdir $(wildcard $(addsuffix /*.cu,$(SRC_DIR)))))
endif