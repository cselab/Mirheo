test: main.cpp
	$(CXX) main.cpp -g -O2 -I$(HDF5_DIR)/include -L$(HDF5_DIR)/lib -lhdf5 -o test

example: example.cpp
	$(CXX) example.cpp -g -O2 -I$(HDF5_DIR)/include -L$(HDF5_DIR)/lib -lhdf5 -o example

clean:
	rm test example

.PHONY = test example
