CC		= g++
NVCC	= nvcc
CCFLAGS = -O4
NVCCFLAGS = -O4 -arch=sm_35
INCLUDE	= 
LIBS	= 

SRC	= $(wildcard *.cu)
EXE	= $(SRC:.cu=.cx)

default : $(EXE)

clean:
	rm $(EXE)	

%.cx : %.cu
	$(NVCC) $(NVCCFLAGS) $(INCLUDE) $< $(LIBS) -o $@
	
