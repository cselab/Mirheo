.PHONY: clean
.SUFFIXES: .o .cpp

CXXFLAGS += -std=c++11  -fopenmp -I../funnel-obstacle/ -O0 -g -DUSE_CUDA
#-pg
CXXFLAGS += -DNDEBUG -Ofast

CUDADPD = ../cuda-dpd-sem/dpd/libcuda-dpd.so
CXXFLAGS += -I$(dir $(CUDADPD))

ifeq "$(gpu)" "1"
CUDADPDCMD := make -C $(dir $(CUDADPD)) $(notdir $(CUDADPD)) 
endif

HEADERS = ../funnel-obstacle/funnel-obstacle.h funnel-bouncer.h particles.h
OBJS = main.o funnel-obstacle.o funnel-bouncer.o particles.o

all: test

particles.o: particles.cpp particles.h
	$(CXX) $(CXXFLAGS) -c $< -o $@

funnel-bouncer.o: funnel-bouncer.cpp funnel-bouncer.h
	$(CXX) $(CXXFLAGS) -c $< -o $@

funnel-obstacle.o: ../funnel-obstacle/*.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

main.o: main.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) -c $< -o $@

test: $(OBJS)
	$(CUDADPDCMD)
	$(CXX) $(CXXFLAGS) $^ $(CUDADPD) -o $@

clean:
	rm -f test *.o *.d

