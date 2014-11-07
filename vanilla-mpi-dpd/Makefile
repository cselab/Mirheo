CXXFLAGS += -std=c++11 -g -O2

OBJS = main.o dpd-interactions.o redistribute-particles.o common.o 

#the this project relies on another folder for the cuda evaluation of the dpd interaction:
#namely CTC/cuda-dpd-sem/dpd

ifeq "$(cuda)" "1"
CXXFLAGS += -D_WITHCUDA_ -I../cuda-dpd-sem/dpd -L/usr/local/cuda-5.5/lib64 -lcudart 
LIBCUDADPD_TARGET = libcuda-dpd
LIBCUDADPD = ../cuda-dpd-sem/dpd/libcuda-dpd.so
endif

test: $(OBJS) $(LIBCUDADPD_TARGET)
	$(CXX) $(CXXFLAGS) $(OBJS) $(LIBCUDADPD)  -o test

%.o: %.cpp %.h
	$(CXX) $(CXXFLAGS) -c $< -o $@

libcuda-dpd: 
	make -C ../cuda-dpd-sem/dpd libcuda-dpd.so

clean:
	rm -f test *.o

#we want to give always a try to make a new libcuda-dpd - too complicated to track dependecies there
.PHONY = clean libcuda-dpd
