PIP ?= python -m pip
CMAKE_FLAGS ?= ""

build:
	mkdir -p build/
	(cd build/;	cmake ${CMAKE_FLAGS} ../)
	make -C build/ -j 12
	cd ..

install: build
	$(PIP) install . --user --upgrade

uninstall:
	$(PIP) uninstall mirheo

docs:
	make -C docs/

test: install
	(cd tests; mir.make test)

clean:; rm -rf build

.PHONY: install uninstall build test clean docs
