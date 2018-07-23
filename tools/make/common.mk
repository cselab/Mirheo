install: $(INST_TARGETS)
	@echo install $^ in $(BIN_PREFIX)
	@mkdir -p $(BIN_PREFIX)
	@for f in $^; do                               \
           cp "$$f" "$(BIN_PREFIX)/$(EXE_PREFIX).$$f"; \
	done
