cmakecache=build/CMakeCache.txt

build: $(cmakecache)
	(cd build; udx.make -j)

$(cmakecache):
	mkdir -p build
	(cd build; . udx.load; cmake ../)

install:
	@(. udx.load; \
	 pip3 install . --user --upgrade)

uninstall:
	pip3 uninstall udevicex

docs:
	make -C docs/
	make -C docs/source/

test:
	(cd tests; udx.make test)

clean:; rm -rf build

.PHONY: install build test clean docs
