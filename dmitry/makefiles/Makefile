config ?= release

.DEFAULT_GOAL := all

.PHONY: md md_cell md_gpu md_gpu_cell clean clean_md clean_md_cell clean_md_gpu clean_md_gpu_cell
 
all: md md_cell md_gpu md_gpu_cell 
   
md:
	$(MAKE) -C cpu_generic config=$(config) extra="$(extra)"
	cp cpu_generic/$@ ./$@
	
md_cell:
	$(MAKE) -C cpu_cell config=$(config) extra="$(extra)"
	cp cpu_cell/$@ ./$@

md_gpu:
	$(MAKE) -C gpu_generic config=$(config) extra="$(extra)"
	cp gpu_generic/$@ ./$@
	
md_gpu_cell:
	$(MAKE) -C gpu_cell config=$(config) extra="$(extra)" 
	cp gpu_cell/$@ ./$@ 

clean_md:
	rm -f cpu_generic/*.o
	rm -f cpu_generic/*.s
	rm -f cpu_generic/*.d
	rm -f cpu_generic/*.cpp
	rm -f cpu_generic/*.cu
	rm -f md cpu_generic/md
	
clean_md_cell:
	rm -f cpu_cell/*.o
	rm -f cpu_cell/*.s
	rm -f cpu_cell/*.d
	rm -f cpu_cell/*.cpp
	rm -f cpu_cell/*.cu
	rm -f md_cell cpu_cell/md_cell
	
clean_md_gpu:
	rm -f gpu_generic/*.o
	rm -f gpu_generic/*.s
	rm -f gpu_generic/*.d
	rm -f gpu_generic/*.cpp
	rm -f gpu_generic/*.cu
	rm -f md_gpu gpu_generic/md_gpu
	
clean_md_gpu_cell:
	rm -f gpu_cell/*.o
	rm -f gpu_cell/*.s
	rm -f gpu_cell/*.d
	rm -f gpu_cell/*.cpp
	rm -f gpu_cell/*.cu
	rm -f md_gpu_cell gpu_cell/md_gpu_cell

clean: clean_md clean_md_cell clean_md_gpu clean_md_gpu_cell
