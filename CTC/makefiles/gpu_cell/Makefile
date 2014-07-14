NVCC = nvcc
FLAGS+= -arch=sm_20 -m 64

config ?= release

ifeq "$(config)" "debug"
FLAGS+= -g -O0 -lineinfo
else ifeq "$(config)" "release"
FLAGS += -O3
else
FLAGS += -O1
endif

LIBS += -lcuda -lcudart -lstdc++
OS = $(shell uname)
ifneq "$(OS)" "Darwin"
LIBS += -lrt
endif

THRUST_TARGET_GPU = -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CUDA
CELL_FLAG = -DMD_USE_CELLLIST

DIRS = ../../source/
CPP_FILES = $(wildcard $(DIRS:%/=%/*.cpp))
OBJ_FILES = $(notdir $(CPP_FILES:.cpp=.o))
INCLUDES += -I$(DIRS)
FLAGS += $(INCLUDES)
FLAGS += $(extra)
FLAGS += $(CELL_FLAG)
	
.DEFAULT_GOAL := md_gpu_cell

ifneq "$(MAKECMDGOALS)" "clean"
-include $(notdir $(patsubst %.cpp,%.d,$(CPP_FILES)))
endif

vpath %.cpp %.h $(DIRS)
    
md_gpu_cell:    $(OBJ_FILES)
		$(NVCC) $(LIBS) -m 64 $^ -o $@

%.o: %.cu
	$(NVCC)  $(FLAGS) $(THRUST_TARGET_GPU) -c $< -o $@
	
%.cu: %.cpp
	cp $< $@
	
%.d: %.cu
	$(NVCC)  $(FLAGS) $(THRUST_TARGET_GPU) --generate-dependencies $< -o $@

