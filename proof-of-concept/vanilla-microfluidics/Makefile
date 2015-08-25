CXXFLAGS += -std=c++11  -fopenmp -I../funnel-obstacle/ -g -Ofast -DNDEBUG
#-DNDEBUG -Ofast

CUDADPD = ../cuda-dpd-sem/dpd/libcuda-dpd.so
CUDASEM = ../cuda-dpd-sem/sem/libcuda-sem.so

CXXFLAGS += -I$(dir $(CUDADPD))
CXXFLAGS += -I$(dir $(CUDASEM))

ifeq "$(gpu)" "1"
CUDADPDCMD := make -C $(dir $(CUDADPD)) $(notdir $(CUDADPD))
CUDASEMCMD := make -C $(dir $(CUDASEM)) $(notdir $(CUDASEM)) 
endif

test: main.cpp ../funnel-obstacle/funnel-obstacle.o
	echo hello
	$(CUDADPDCMD)
	$(CUDASEMCMD)
	$(CXX) $(CXXFLAGS) $^ -L$(dir $(CUDADPD)) -L$(dir $(CUDASEM)) \
	-lcuda-dpd -lcuda-sem -o test
#

../funnel-obstacle/funnel-obstacle.o: ../funnel-obstacle/*.cpp
	make -C ../funnel-obstacle funnel-obstacle.o


clean:
	rm -f test

.PHONY = clean
