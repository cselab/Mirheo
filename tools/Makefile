include config/install

install: install.modules \
	 install.compile \
	 install.run \
	 install.postprocess

test: test.postprocess \
      test.run


install.modules:;     @(cd modules;     make install)
install.compile:;     @(cd compile;     make install)
install.run:;         @(cd run;         make install)
install.postprocess:; @(cd postprocess; make install)

test.postprocess:;    @(cd postprocess; make test)
test.run:;            @(cd run;         make test)

.PHONY: install install.modules install.postprocess \
	test test.postprocess

