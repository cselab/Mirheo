CXXFLAGS += -std=c++11 -g -O2

ifeq "$(cuda)" "1"
CXXFLAGS += -D_WITHCUDA_ -I../cuda-dpd-sem/dpd -L/usr/local/cuda-5.5/lib64 -lcudart 
LIBCUDADPD_TARGET = libcuda-dpd
LIBCUDADPD = ../cuda-dpd-sem/dpd/libcuda-dpd.so
endif

test: main.o dpd-interactions.o redistribute-particles.o common.o $(LIBCUDADPD_TARGET)
	$(CXX) $(CXXFLAGS) main.o dpd-interactions.o redistribute-particles.o common.o $(LIBCUDADPD)  -o test

%.o: %.cpp %.h
	$(CXX) $(CXXFLAGS) -c $< -o $@

libcuda-dpd: 
	make -C ../cuda-dpd-sem/dpd libcuda-dpd.so

clean:
	rm -f test

.PHONY = clean libcuda-dpd
