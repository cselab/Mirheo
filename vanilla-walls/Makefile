CXXFLAGS += -std=c++11 -g -Ofast -fopenmp -I../funnel-obstacle/

ifeq "$(gpu)" "1"
CUDADPD = ../cuda-dpd-sem/dpd/libcuda-dpd.so
CXXFLAGS += -I$(dir $(CUDADPD))
CUDADPDCMD := make -C $(dir $(CUDADPD)) $(notdir $(CUDADPD)) 
endif

test: main.cpp ../funnel-obstacle/funnel-obstacle.o
	echo hello
	$(CUDADPDCMD)
	$(CXX) $(CXXFLAGS) $^ $(CUDADPD) -o test

../funnel-obstacle/funnel-obstacle.o: ../funnel-obstacle/*.cpp
	make -C ../funnel-obstacle funnel-obstacle.o


clean:
	rm -f test

.PHONY = clean
