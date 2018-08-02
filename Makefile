build:
	mkdir -p build/
	(cd build/;	cmake ../)
	make -C build/ -j
	cd ..

install: build
	pip install . --user --upgrade

uninstall:
	pip uninstall udevicex

docs:
	make -C docs/
	make -C docs/source/

test: install
	(cd tests; udx.make test)

clean:; rm -rf build

.PHONY: install build test clean docs
