NVCC = nvcc
NVCCFLAGS = -std=c++11 -arch=compute_61 -code=sm_61 -O3 -g -lineinfo --expt-extended-lambda -Xcompiler " -fopenmp"
#-O0 -G -Xptxas -O0

LIBS = -lmpi

SRC_DIR = ../core ./ ../core/xml
VPATH = $(addsuffix :,$(SRC_DIR))
NVCCFLAGS += -I../

OBJ_DIR = obj

wall_obj = scan.o celllist.o halo_exchanger.o dpd.o test_wall.o redistributor.o wall.o integrate.o
onerank_obj = scan.o celllist.o halo_exchanger.o dpd.o test_onerank.o redistributor.o integrate.o
manyranks_obj = scan.o celllist.o halo_exchanger.o dpd.o test_manyranks.o redistributor.o integrate.o
halo_obj    = scan.o celllist.o halo_exchanger.o test_halo.o
redist_obj  = scan.o celllist.o redistributor.o test_redist.o
log_obj     = test_logger.o
cells_obj   = scan.o celllist.o test_cells.o pugixml.o components.o integrate.o interactions.o wall.o
dpd_obj     = scan.o celllist.o test_dpd.o pugixml.o components.o integrate.o interactions.o wall.o


default:
	@echo "Please specify a target!"
	
test_wall: $(addprefix $(OBJ_DIR)/,$(wall_obj))
	$(NVCC) $(NVCCFLAGS) $(LIBS) $^ -o $@

test_cells: $(addprefix $(OBJ_DIR)/,$(cells_obj))
	$(NVCC) $(NVCCFLAGS) $(LIBS) $^ -o $@

test_logger: $(addprefix $(OBJ_DIR)/,$(log_obj))
	$(NVCC) $(NVCCFLAGS) $(LIBS) $^ -o $@

test_onerank: $(addprefix $(OBJ_DIR)/,$(onerank_obj))
	$(NVCC) $(NVCCFLAGS) $(LIBS) $^ -o $@
	
test_manyranks: $(addprefix $(OBJ_DIR)/,$(manyranks_obj))
	$(NVCC) $(NVCCFLAGS) $(LIBS) $^ -o $@
	
test_halo: $(addprefix $(OBJ_DIR)/,$(halo_obj))
	$(NVCC) $(NVCCFLAGS) $(LIBS) $^ -o $@
	
test_redist: $(addprefix $(OBJ_DIR)/,$(redist_obj))
	$(NVCC) $(NVCCFLAGS) $(LIBS) $^ -o $@
	
test_dpd: $(addprefix $(OBJ_DIR)/,$(dpd_obj))
	$(NVCC) $(NVCCFLAGS) $(LIBS) $^ -o $@
	

$(OBJ_DIR)/%.d: %.cu
	$(NVCC) $(NVCCFLAGS) -M $< > $(patsubst %.d,%.TMPd,$@)
	@mv $(patsubst %.d,%.TMPd,$@) $@
	
$(OBJ_DIR)/%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@
	
clean:
	rm -f *.o *.d test_onerank test_halo test_logger test_cells test_dpd test_redist
	
.PHONY: clean default


ifneq "$(MAKECMDGOALS)" "clean"
-include $(patsubst %.cu,$(OBJ_DIR)/%.d,$(notdir $(wildcard $(addsuffix /*.cu,$(SRC_DIR)))))
endif
