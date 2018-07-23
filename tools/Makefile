include config

install: install.modules \
	 install.postprocess

test: test.postprocess

install.modules:
	(cd modules; make install)

install.postprocess:
	(cd postprocess; make install)
test.postprocess:
	(cd postprocess; make test)

.PHONY: install install.modules install.postprocess \
	test test.postprocess

