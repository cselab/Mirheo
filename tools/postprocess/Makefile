INST_TARGETS = avgh5 avgh5.py

include ../config
include ../make/common.mk

avgh5: avgh5.in
	sed "s/@EXE_PREFIX@/$(EXE_PREFIX)/" $< > $@
	chmod +x $@

test:; @atest `find test -type f`

.PHONY: install test
