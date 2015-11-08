# Generate microfluidic geometry

Set of scripts to generate device geometries as Signed Distance Function in \*.dat format.
The dat format consists of header and the binary float data:
```
<xSize> <ySize> <zSize>
<xGridSize> <yGridSize> <zGridSize>
<Binary data>
```
Where size is the geometry length units (typically microns), grid size defines how many grid points are there.

## Parabolic funnels
Geometry mimicing the microfluidic device by McFaul et al [Cell separation based on size and deformability using microfluidic funnel ratchets](http://www.ncbi.nlm.nih.gov/pubmed/22517056)
To generate a this geometry with 10 rows and 20 columns and with walls in z direction of width 4:
```
cd funnels
make
./funnel -nColumns=20 -nRows=10 -zMargin=4 -out=geom.dat
```

## Later displacement device
Geometry reproducing CTC-iChip1 module by Karabacak et al [Microfluidic, marker-free isolation of circulating tumor cells from blood samples](http://www.nature.com/nprot/journal/v9/n3/full/nprot.2014.044.html)

To build geometry constisting of 13 columns and 59 rows repeated twice, with wall widht 2 and such grid resolution that 0.5 grid points correspond to 1 unit of length:
```
cd ctc-ichip
make 
./ctc-ichip -nColumns=13 -nRows=59 -nRepeat=2 -zMargin=2.0 -out=13x59x2-05.dat -zResolution=0.5
```

