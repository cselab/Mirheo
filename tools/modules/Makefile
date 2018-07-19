include ../config

src = load load.falcon

install: $(src)
	@echo install $^ in $(BIN_PREFIX)
	@mkdir -p $(BIN_PREFIX)
	@for f in $(src); do                          \
           cp "$$f" "$(BIN_PREFIX)/$(EXE_PREFIX).$$f"; \
	done

load: load.in
	sed "s/@EXE_PREFIX@/$(EXE_PREFIX)/" $< > $@
