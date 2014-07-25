#pragma once

void build_clists(float * const device_xyzuvw, int np, const float rc, 
		      const int xcells, const int ycells, const int zcells,
		      const float xdomainstart, const float ydomainstart, const float zdomainstart,
		      int * const host_order, int * device_cellsstart, int * device_cellsend);
		  
