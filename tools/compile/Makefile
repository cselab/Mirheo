INST_TARGETS = make make.flags.falcon make.flags.daint

include ../config
include ../make/common.mk

make: make.in
	sed "s/@EXE_PREFIX@/$(EXE_PREFIX)/" $< > $@
	chmod +x $@
