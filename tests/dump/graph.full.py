#!/usr/bin/env python

import mirheo as mir

ranks  = (1, 1, 1)
domain = (4, 4, 4)

u = mir.Mirheo(ranks, domain, debug_level=3, log_filename='log', no_splash=True)

u.save_dependency_graph_graphml("tasks.full", current=False)

# sTEST: dump.graph.full
# cd dump
# rm -rf tasks.graphml
# mir.run --runargs "-n 1" ./graph.full.py
# cat tasks.full.graphml > tasks.out.txt

