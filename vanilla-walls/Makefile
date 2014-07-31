CXXFLAGS += -std=c++11 -g -Ofast -fopenmp

ifeq "$(gpu)" "1"
CUDADPD = ../cuda-dpd-sem/dpd/libcuda-dpd.so
CXXFLAGS += -I$(dir $(CUDADPD))
CUDADPDCMD := make -C $(dir $(CUDADPD)) $(notdir $(CUDADPD)) 
endif

test: main.cpp
	echo hello
	$(CUDADPDCMD)
	$(CXX) $(CXXFLAGS) main.cpp $(CUDADPD) -o test



clean:
	rm -f test

.PHONY = clean
