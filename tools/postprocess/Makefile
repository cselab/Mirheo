INST_TARGETS = avgh5 avgh5.py post

include ../config/install
include ../make/common.mk

%: %.in
	sed "s/@EXE_PREFIX@/$(EXE_PREFIX)/" $< > $@
	chmod +x $@

test:; @atest `find test -type f`

.PHONY: install test
