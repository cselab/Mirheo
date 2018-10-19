INST_TARGETS = run

include ../config/install
include ../make/common.mk

%: %.in
	sed "s/@EXE_PREFIX@/$(EXE_PREFIX)/" $< | \
	sed "s/@RUN_CMD@/$(RUN_CMD)/"  > $@> $@
	chmod +x $@

test:; atest `find tests -type f`

.PHONY: test
