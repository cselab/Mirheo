CC = g++
CPPFLAGS+= -fopenmp -std=c++11 -Wno-unknown-pragmas -Wno-reorder -Wno-comment -Wno-deprecated

config ?= release

ifeq "$(config)" "debug"
CPPFLAGS+= -O0 -g
else ifeq "$(config)" "release"
CPPFLAGS += -O3 -g -fno-tree-vectorize -funroll-loops -DNDEBUG
else
CPPFLAGS += -O1 -g
endif

LIBS += #-lstdc++
OS = $(shell uname)
ifneq "$(OS)" "Darwin"
LIBS += -lrt
endif

THRUST_TARGET_CPU = -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_BACKEND_OMP
THRUST_TARGET_GPU = -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_BACKEND_CUDA

DIRS = ../source/
CPP_CUDA_FILES = $(DIRS)Interaction.cpp
CPP_FILES = $(filter-out $(CPP_CUDA_FILES), $(wildcard $(DIRS:%/=%/*.cpp)))
OBJ_FILES = $(notdir $(CPP_FILES:.cpp=.o))
CUDA_OBJ_FILES = $(notdir $(CPP_CUDA_FILES:.cpp=.cudaobj))
CPP_OBJ_FILES = $(notdir $(CPP_CUDA_FILES:.cpp=.o))
INCLUDES += -I$(DIRS) -I../../cuda-dpd-sem/

	
.DEFAULT_GOAL := ctc
.PHONY: clean

ifneq "$(MAKECMDGOALS)" "clean"
-include $(notdir $(patsubst %.cpp,%.d,$(wildcard $(DIRS:%/=%/*.cpp))))
endif

vpath %.cpp %.h %.cu $(DIRS)

clean:
	rm -f *.o
	rm -f *.d
	rm -f *.cudaobj
	rm -f ctc
	rm -f ctcgpu
    
ctc: $(OBJ_FILES) $(CPP_OBJ_FILES)
	$(CC) $^  $(LIBS) -fopenmp -o $@
	
ctcgpu: $(OBJ_FILES) $(CUDA_OBJ_FILES)
	$(CC) $(GPULIBS) -fopenmp $^ -o $@

$(OBJ_FILES):%.o:%.cpp
	$(CC)  $(CPPFLAGS) $(THRUST_TARGET_CPU) $(INCLUDES) -c $< -o $@
	
$(CPP_OBJ_FILES):%.o:%.cpp
	$(CC)  $(CPPFLAGS) $(THRUST_TARGET_CPU) $(INCLUDES) -c $< -o $@
	
$(CUDA_OBJ_FILES):%.cudaobj:%.cpp $(DIRS)Gpukernels.h
	$(CC) $(CPPFLAGS) -D__GPU__ $(THRUST_TARGET_CPU) $(INCLUDES) -c $< -o $@
	
%.d: %.cpp
	$(CC)  $(CPPFLAGS) $(THRUST_TARGET_CPU) -MM $< > $*.d
	
