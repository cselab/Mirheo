.PHONY: clean
.SUFFIXES: .o .cpp

# introduced 2 options - with gpu support -DUSE_CUDA
#					   - with gpu support and SEM -DUSE_SEM

CXXFLAGS += -std=c++11  -fopenmp -I../funnel-obstacle/ -O0 -g 

ifeq "$(gpu)" "1"
CXXFLAGS += -DUSE_CUDA
CXXFLAGS += -I$(dir $(CUDADPD))
CUDADPD = ../cuda-dpd-sem/dpd/libcuda-dpd.so
CUDADPDCMD := make -C $(dir $(CUDADPD)) $(notdir $(CUDADPD)) 

ifeq "$(sem)" "1"
CXXFLAGS += -DUSE_SEM
CXXFLAGS += -I$(dir $(CUDASEM))
CUDASEM = ../cuda-dpd-sem/sem/libcuda-sem.so
CUDASEMCMD := make -C $(dir $(CUDASEM)) $(notdir $(CUDASEM))
endif

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
	$(CUDASEMCMD)
	$(CXX) $(CXXFLAGS) $^ $(CUDADPD) $(CUDASEM) -o $@

clean:
	rm -f test *.o *.d

