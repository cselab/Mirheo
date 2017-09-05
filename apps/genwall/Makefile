NVCC = nvcc
NVCCFLAGS = -std=c++11 -arch=compute_61 -code=sm_61 -g -lineinfo --expt-extended-lambda -Xcompiler "-fopenmp"
NOWARNFLAG = -Xcompiler "-Wno-cpp"

HDF5_INC = /usr/include/hdf5/mpich/
HDF5_LIB = /usr/lib/x86_64-linux-gnu/

LIBS = -lmpi -lhdf5_mpich -L$(HDF5_LIB)

SRC_DIR = ../../core/ ../../core/interactions ../../core/pvs ../../core/xml ../../core/mpi ../../plugins ./
BIN_DIR = bin
DEBUG_DIR = debug

VPATH = $(addsuffix :,$(SRC_DIR))
NVCCFLAGS += -I../../ -I$(HDF5_INC)
NVCC_RELEASE_FLAGS = $(NVCCFLAGS) -O3
NVCC_DEBUG_FLAGS = $(NVCCFLAGS) -G -Xcompiler "-O0"

objects = main.o scan.o celllist.o wall.o sampler.o simulation.o pugixml.o task_scheduler.o initial_conditions.o argument_parser.o\
	particle_vector.o object_vector.o \
	particle_exchanger.o particle_halo_exchanger.o particle_redistributor.o \
	write_xdmf.o dumpavg.o stats.o temperaturize.o \
	freeze_particles.o


# Release binary
genwall: $(addprefix $(BIN_DIR)/,$(objects))
	$(NVCC) $(NVCC_RELEASE_FLAGS) $(LIBS) $^ -o $@

$(BIN_DIR)/%.d: %.cu
	@mkdir -p $(BIN_DIR)
	$(NVCC) $(NVCC_RELEASE_FLAGS) $(NOWARNFLAG) -M $< > $(patsubst %.d,%.TMPd,$@)
	@printf "$(BIN_DIR)/" | cat - $(patsubst %.d,%.TMPd,$@)  > $@ && rm $(patsubst %.d,%.TMPd,$@)
	@#| perl -n -e ' if (/\s+\./) { s/(\s+)\./$$1..\/./; } print; '
	
$(BIN_DIR)/%.o: %.cu
	$(NVCC) $(NVCC_RELEASE_FLAGS) -c $< -o $@
	
clean_release:
	rm -f $(BIN_DIR)/*
	@test -e $(BIN_DIR) && rmdir -p $(BIN_DIR); true
	rm -f poiseuille


# Debug binary
genwall_debug: $(addprefix $(DEBUG_DIR)/,$(objects))
	$(NVCC) $(NVCC_DEBUG_FLAGS) $(LIBS) $^ -o $@

$(DEBUG_DIR)/%.d: %.cu
	@mkdir -p $(DEBUG_DIR)
	$(NVCC) $(NVCC_DEBUG_FLAGS) $(NOWARNFLAG) -M $< > $(patsubst %.d,%.TMPd,$@)
	@printf "$(DEBUG_DIR)/" | cat - $(patsubst %.d,%.TMPd,$@)  > $@ && rm $(patsubst %.d,%.TMPd,$@)
	@#| perl -n -e ' if (/\s+\./) { s/(\s+)\./$$1..\/./; } print; '
	
$(DEBUG_DIR)/%.o: %.cu
	$(NVCC) $(NVCC_DEBUG_FLAGS) -c $< -o $@

clean_debug:
	rm -f $(DEBUG_DIR)/*
	@test -e $(DEBUG_DIR) && rmdir -p $(DEBUG_DIR); true
	rm -f poiseuille_debug
	
	

clean: clean_release clean_debug

	
.PHONY: clean clean_release clean_debug default


ifeq (,$(findstring clean,$(MAKECMDGOALS)))
-include $(patsubst %.cu,$(BIN_DIR)/%.d,  $(notdir $(wildcard $(addsuffix /*.cu,$(SRC_DIR)))))
-include $(patsubst %.cu,$(DEBUG_DIR)/%.d,$(notdir $(wildcard $(addsuffix /*.cu,$(SRC_DIR)))))
endif
