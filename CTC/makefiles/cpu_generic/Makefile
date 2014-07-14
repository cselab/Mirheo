CC = g++
NVCC = nvcc
CPPFLAGS+= -g -std=c++11 -ftree-vectorize -funroll-loops -Wno-unknown-pragmas -Wno-reorder -Wno-comment -Wno-deprecated

viz ?= 0
config ?= release

ifeq "$(config)" "debug"
CPPFLAGS+= -O0
else ifeq "$(config)" "release"
CPPFLAGS += -O3
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

CPPFLAGS += $(INCLUDES)

DIRS = ../../source/
CPP_FILES = $(wildcard $(DIRS:%/=%/*.cpp))
OBJ_FILES = $(notdir $(CPP_FILES:.cpp=.o))
INCLUDES += -I$(DIRS)

	
.DEFAULT_GOAL := md

ifneq "$(MAKECMDGOALS)" "clean"
-include $(notdir $(patsubst %.cpp,%.d,$(CPP_FILES)))
endif

vpath %.cpp %.h $(DIRS)
    
md: $(OBJ_FILES)
	$(CC) $(LIBS) $^ -o $@

%.o: %.cpp
	$(CC)  $(CPPFLAGS) $(THRUST_TARGET_CPU) -c $< -o $@
	
%.d: %.cpp
	$(CC)  $(CPPFLAGS) $(THRUST_TARGET_CPU) -MD -c $<

