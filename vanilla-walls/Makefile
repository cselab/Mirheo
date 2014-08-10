.PHONY: clean
.SUFFIXES: .o .cpp

CXXFLAGS += -std=c++11  -fopenmp -I../funnel-obstacle/ -O0 -g -DUSE_CUDA

#-DNDEBUG -Ofast

CUDADPD = ../cuda-dpd-sem/dpd/libcuda-dpd.so
CXXFLAGS += -I$(dir $(CUDADPD))

ifeq "$(gpu)" "1"
CUDADPDCMD := make -C $(dir $(CUDADPD)) $(notdir $(CUDADPD)) 
endif

test: main.cpp funnel-obstacle.o
	echo hello
	$(CUDADPDCMD)
	$(CXX) $(CXXFLAGS) $^ $(CUDADPD) -o test

%.o:%.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $<

clean:
	rm -f test *.o *.d

