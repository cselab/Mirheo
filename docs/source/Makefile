files := $(addsuffix .gen,$(wildcard user/*.rst))

.DEFAULT_GOAL := all

prepare:
	rm -rf ./temp
	rm user/*.rst.gen
	python gen_autodoc.py ./ ./temp

user/%.rst.gen: user/%.rst prepare
	touch $<
	python gen_autodoc.py ./ ./temp $<
	cat $< $<.tmp | perl -p -e 's/automodule::/comment/g' > $@
	rm $<.tmp

all : $(files)
	echo Done!
