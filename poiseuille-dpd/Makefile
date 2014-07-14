CXXFLAGS += -I../cuda-dpd/ -std=c++11 -L/usr/local/cuda-5.5/lib64/ -lcudart -lcurand

test:	main.cpp ../cuda-dpd/cuda-dpd.so
	$(CXX) $(CXXFLAGS) $^ -o test

../cuda-dpd/cuda-dpd.so:
	$(MAKE) -C ../cuda-dpd cuda-dpd.so
clean:
	$(MAKE) -C ../cuda-dpd clean
	rm -f test

.PHONY = clean ../cuda-dpd/cuda-dpd.so
