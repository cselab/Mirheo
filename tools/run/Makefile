include ../config

src = run

install: $(src)
	@echo install $^ in $(BIN_PREFIX)
	@mkdir -p $(BIN_PREFIX)
	@for f in $(src); do                          \
           cp "$$f" "$(BIN_PREFIX)/$(EXE_PREFIX).$$f"; \
	done

%: %.in
	sed "s/@EXE_PREFIX@/$(EXE_PREFIX)/" $< | \
	sed "s/@RUN_CMD@/$(RUN_CMD)/"  > $@> $@
	chmod +x $@
