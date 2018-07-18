UDEVICEX_PATH = ../..

binary = dpd

# Targets
objects = test_dpd.o celllist.o uniform_ic.o particle_vector.o object_vector.o pairwise.o stacktrace_explicit.o 

CXXFLAGS += -Xcompiler "-fopenmp"
LDFLAGS  += -Xcompiler "-fopenmp"
	
include ../../makefiles/Makefile.base
