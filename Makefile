CMAKE ?= cmake
PIP ?= python -m pip
CMAKE_FLAGS ?= ""

.PHONY: build install compile_and_copy uninstall docs test clean

build:
	mkdir -p build
	(cd build && ${CMAKE} ${CMAKE_FLAGS} -DBUILD_TESTS=OFF ../)
	(cd build && $(MAKE))

# https://stackoverflow.com/questions/1871549/determine-if-python-is-running-inside-virtualenv
# The --user argument is given only if we are not inside a virtualenv.
install: build
	$(PIP) install . $(shell python -c "import sys; hasattr(sys, 'real_prefix') or print('--user')") --upgrade

uninstall:
	$(PIP) uninstall mirheo -y

docs:
	$(MAKE) -C docs/

test: install
	(cd tests && mir.make test)

units:
	mkdir -p build
	(cd build && ${CMAKE} ${CMAKE_FLAGS} -DBUILD_TESTS=ON ../)
	(cd build && $(MAKE))
	(cd build && $(MAKE) test)

clean:
	rm -rf build

.PHONY: install uninstall build test clean docs units
