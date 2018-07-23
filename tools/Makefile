include config

install: install.modules \
	 install.postprocess

install.modules:
	(cd modules; make install)

install.postprocess:
	(cd postprocess; make install)

.PHONY: install install.modules
