#!/usr/bin/env python

import sys
import numpy
import h5py
import re

init = 0
avg = dict()

fout = h5py.File(sys.argv[-1], 'w')

for fname in sys.argv[1:-1]:
	print fname

	f = h5py.File(fname, 'r')
	datasets = [f[item] for item in f.keys()]
		
	if init == 0:
		avgSets = [fout.create_dataset(ds.name, ds.shape, ds.dtype) for ds in datasets]
		init = 1
	
	for ds, a in zip(datasets, avgSets):
		a[:] += ds[:]
		
for a in avgSets:
	a[:] /= (len(sys.argv) - 2)


foutShortName = re.match('(.*/|^)(.*)', sys.argv[-1]).group(2)
fxdmfName = re.sub('\..*$', '.xmf', sys.argv[-1])
fxdmf = open(fxdmfName, 'w')
datastr = ''

for a in avgSets:

	type="Scalar"
	lastdim=1
	if len(a.shape) > 3 and a.shape[3] == 3:
		type="Vector"
		lastdim=3

	if len(a.shape) > 3 and a.shape[3] == 6:
		type="Tensor6"
		lastdim=6


	datastr += '''			<Attribute Name="%s" AttributeType="%s" Center="Cell">
				<DataItem Dimensions="%d %d %d %d" NumberType="Float" Precision="4" Format="HDF">
					%s:%s
				</DataItem>
			</Attribute>
''' % (str(a.name)[1:], type, a.shape[0], a.shape[1], a.shape[2], lastdim, foutShortName, str(a.name))

s = '''<?xml version="1.0" ?>
<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" []>
<Xdmf Version="2.0">
	<Domain>
		<Grid Name="mesh" GridType="Uniform">
			<Topology TopologyType="3DCORECTMesh" Dimensions="%d %d %d"/>
			<Geometry GeometryType="ORIGIN_DXDYDZ">
				<DataItem Name="Origin" Dimensions="3" NumberType="Float" Precision="4" Format="XML">
					0.000000e+00 0.000000e+00 0.000000e+00
				</DataItem>
				<DataItem Name="Spacing" Dimensions="3" NumberType="Float" Precision="4" Format="XML">
					1.0 1.0 1.0
				</DataItem>
			</Geometry>
%s
		</Grid>
	</Domain>
</Xdmf>''' % (avgSets[0].shape[0] + 1, avgSets[0].shape[1] + 1, avgSets[0].shape[2] + 1, datastr)

fout.close()
fxdmf.write(s)
fxdmf.close()
