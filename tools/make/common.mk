install: $(INST_TARGETS)
	@mkdir -p $(BIN_PREFIX)
	@for f in $^; do                                    \
	   echo install $(EXE_PREFIX).$$f in $(BIN_PREFIX); \
           cp "$$f" "$(BIN_PREFIX)/$(EXE_PREFIX).$$f";      \
	done
