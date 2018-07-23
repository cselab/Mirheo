install:
	pip3 install . --user --upgrade

test:
	(cd tests; make test)
