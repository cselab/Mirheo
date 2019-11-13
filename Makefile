PIP ?= python -m pip
CMAKE_FLAGS ?= ""

.PHONY: build install compile_and_copy uninstall docs test clean

build:
	mkdir -p build/
	(cd build/ && cmake ${CMAKE_FLAGS} ../)
	$(MAKE) -C build/
	cd ..

# https://stackoverflow.com/questions/1871549/determine-if-python-is-running-inside-virtualenv
# The --user argument is given only if we are not inside a virtualenv.
install: build
	$(PIP) install . $(shell python -c "import sys; hasattr(sys, 'real_prefix') or print('--user')") --upgrade

# This only compiles and overwrites the already installed `.so` file. It skips
# the `cmake` step and shortens the long `pip install . --upgrade` step. This
# won't detect new files or any changes to `.py` files.
#
# To automatically import the local Python code, add this to your virtualenv bin/activate:
# 	export PATH=full-path-to-mirheo-repository-root:$PATH
compile_and_copy:
	(cd build && $(MAKE))
	cp $(shell python -c "import os, mirheo; p = mirheo._libmirheo_file; print(os.path.join('build', os.path.basename(p)), p)")

uninstall:
	$(PIP) uninstall mirheo -y

docs:
	$(MAKE) -C docs/

test: install
	(cd tests && mir.make test)

clean:
	rm -rf build

.PHONY: install uninstall build test clean docs
