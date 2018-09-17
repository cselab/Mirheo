include config

install: install.utils \
	 install.modules \
	 install.compile \
	 install.run \
	 install.postprocess

test: test.postprocess \
      test.run


install.utils:;       (cd utils;       make install)
install.modules:;     (cd modules;     make install)
install.compile:;     (cd compile;     make install)
install.run:;         (cd run;         make install)
install.postprocess:; (cd postprocess; make install)

test.postprocess:;    (cd postprocess; make test)
test.run:;            (cd run;         make test)

.PHONY: install install.utils install.modules install.postprocess \
	test test.postprocess

