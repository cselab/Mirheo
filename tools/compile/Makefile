INST_TARGETS = make

include ../config
include ../make/common.mk

make: make.in
	sed "s/@EXE_PREFIX@/$(EXE_PREFIX)/" $< > $@
	chmod +x $@
