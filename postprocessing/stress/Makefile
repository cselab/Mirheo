CXX = mpicxx

stress: main.cpp
	$(CXX) main.cpp -I../ -g -O3 -Wno-deprecated-declarations -Wno-unused-result -o stress

clean:
	rm -f stress

.PHONY = clean
