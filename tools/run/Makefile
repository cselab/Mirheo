INST_TARGETS = run

include ../config
include ../make/common.mk

%: %.in
	sed "s/@EXE_PREFIX@/$(EXE_PREFIX)/" $< | \
	sed "s/@RUN_CMD@/$(RUN_CMD)/"  > $@> $@
	chmod +x $@
