CXXFLAGS += -I../cuda-dpd-sem/dpd -std=c++11 -L/usr/local/cuda-5.5/lib64/ -L../cuda-dpd-sem/dpd -lcudart -lcurand

test:	main.cpp libcuda-dpd.so Makefile
	$(CXX) $(CXXFLAGS) main.cpp -lcuda-dpd -o test

libcuda-dpd.so:
	$(MAKE) -C ../cuda-dpd-sem/dpd libcuda-dpd.so
clean:
	$(MAKE) -C ../cuda-dpd-sem/dpd clean
	rm -f test

.PHONY = clean libcuda-dpd.so
