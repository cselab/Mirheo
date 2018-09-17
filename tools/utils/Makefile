INST_TARGETS = host

include ../config
include ../make/common.mk

%: %.in
	sed "s/@EXE_PREFIX@/$(EXE_PREFIX)/" $< | \
	sed "s/@RUN_CMD@/$(RUN_CMD)/"  > $@> $@
	chmod +x $@
