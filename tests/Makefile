NVCC = nvcc
NVCCFLAGS = -std=c++11 -arch=compute_61 -code=sm_61 -O3 -g -lineinfo --expt-extended-lambda -Xcompiler "-fopenmp -march=native"
#-O0 -G -Xptxas -O0

LIBS = -lmpi

SRC_DIR = ../core ./ ../core/xml ../core/mpi
VPATH = $(addsuffix :,$(SRC_DIR))
NVCCFLAGS += -I../

OBJ_DIR = obj

wall_obj =    scan.o celllist.o test_wall.o pugixml.o components.o integrate.o interactions.o wall.o halo_exchanger.o redistributor.o
onerank_obj = scan.o celllist.o halo_exchanger.o dpd.o test_onerank.o redistributor.o integrate.o
manyranks_obj = scan.o celllist.o test_manyranks.o pugixml.o components.o integrate.o interactions.o wall.o halo_exchanger.o redistributor.o
halo_obj    = scan.o celllist.o test_halo.o pugixml.o components.o integrate.o interactions.o wall.o particle_exchanger.o particle_halo_exchanger.o
obj_halo_obj= scan.o celllist.o test_obj_halo.o pugixml.o components.o integrate.o interactions.o wall.o object_halo_exchanger.o object_vector.o particle_exchanger.o
bounce_obj  = scan.o celllist.o test_bounce.o pugixml.o components.o integrate.o interactions.o wall.o object_halo_exchanger.o object_vector.o particle_exchanger.o
redist_obj  = scan.o celllist.o test_redist.o particle_exchanger.o particle_redistributor.o components.o pugixml.o wall.o interactions.o integrate.o
log_obj     = test_logger.o
cells_obj   = scan.o celllist.o test_cells.o pugixml.o components.o integrate.o interactions.o wall.o
dpd_obj     = scan.o celllist.o test_dpd.o pugixml.o components.o integrate.o interactions.o wall.o
scheduler_obj = task_scheduler.o test_scheduler.o


default:
	@echo "Please specify a target!"
	
test_wall: $(addprefix $(OBJ_DIR)/,$(wall_obj))
	$(NVCC) $(NVCCFLAGS) $(LIBS) $^ -o $@

test_bounce: $(addprefix $(OBJ_DIR)/,$(bounce_obj))
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

test_obj_halo: $(addprefix $(OBJ_DIR)/,$(obj_halo_obj))
	$(NVCC) $(NVCCFLAGS) $(LIBS) $^ -o $@
	
test_redist: $(addprefix $(OBJ_DIR)/,$(redist_obj))
	$(NVCC) $(NVCCFLAGS) $(LIBS) $^ -o $@
	
test_dpd: $(addprefix $(OBJ_DIR)/,$(dpd_obj))
	$(NVCC) $(NVCCFLAGS) $(LIBS) $^ -o $@
	
test_scheduler: $(addprefix $(OBJ_DIR)/,$(scheduler_obj))
	$(NVCC) $(NVCCFLAGS) $(LIBS) $^ -o $@
	

$(OBJ_DIR)/%.d: %.cu
	@mkdir -p $(OBJ_DIR)
	$(NVCC) $(NVCCFLAGS) -M $< > $(patsubst %.d,%.TMPd,$@)
	@printf "$(OBJ_DIR)/" | cat - $(patsubst %.d,%.TMPd,$@)  > $@ && rm $(patsubst %.d,%.TMPd,$@)
	
$(OBJ_DIR)/%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@
	
clean:
	rm -f obj/* test_onerank test_halo test_logger test_cells test_dpd test_redist
	
.PHONY: clean default


ifneq "$(MAKECMDGOALS)" "clean"
-include $(patsubst %.cu,$(OBJ_DIR)/%.d,$(notdir $(wildcard $(addsuffix /*.cu,$(SRC_DIR)))))
endif
