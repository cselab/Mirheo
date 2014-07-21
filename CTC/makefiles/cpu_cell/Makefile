CC = g++
NVCC = nvcc
CPPFLAGS+= -fopenmp -g -std=c++11 -Wno-unknown-pragmas -Wno-reorder -Wno-comment -Wno-deprecated

config ?= release

ifeq "$(config)" "debug"
CPPFLAGS+= -O0
else ifeq "$(config)" "release"
CPPFLAGS += -O3 -fno-tree-vectorize
else
CPPFLAGS += -O1
endif

#OPENMP_FLAG= -fopenmp

LIBS += -lstdc++
OS = $(shell uname)
ifneq "$(OS)" "Darwin"
LIBS += -lrt
endif

THRUST_TARGET_CPU = -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_BACKEND_OMP
CELL_FLAG = -DMD_USE_CELLLIST

CPPFLAGS += $(INCLUDES)
CPPFLAGS += $(CELL_FLAG)

DIRS = ../../source/
CPP_FILES = $(wildcard $(DIRS:%/=%/*.cpp))
OBJ_FILES = $(notdir $(CPP_FILES:.cpp=.o))
INCLUDES += -I$(DIRS)

	
.DEFAULT_GOAL := md_cell

ifneq "$(MAKECMDGOALS)" "clean"
-include $(notdir $(patsubst %.cpp,%.d,$(CPP_FILES)))
endif

vpath %.cpp %.h $(DIRS)
    
md_cell: $(OBJ_FILES)
	$(CC) $(LIBS) -fopenmp $^ -o $@

%.o: %.cpp
	$(CC)  $(CPPFLAGS) $(THRUST_TARGET_CPU) -c $< -o $@
	
%.d: %.cpp
	$(CC)  $(CPPFLAGS) $(THRUST_TARGET_CPU) -MD -c $<
	
