cmakecache=build/CMakeCache.txt

build: $(cmakecache)
	(cd build; make -j)

$(cmakecache):
	mkdir -p build
	(cd build; cmake ../)

install:
	pip3 install . --user --upgrade

test:
	(cd tests; make test)

clean:; rm -rf build

.PHONY: install build test clean
