#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define MAX_NODES	20000
#ifndef min
#define min(a,b) ((a)<(b)?(a):(b))
#endif

struct xyz_info
{
	int r;
	int x, y, z;
};


struct xyz_info xyz[MAX_NODES];
struct xyz_info xyzsize;

int dist(int i, int j)
{
	struct xyz_info a, b;
	a = xyz[i];
	b = xyz[j];

	int a_x = a.x;
	int b_x = b.x;
	int a_y = a.y;
	int b_y = b.y;
	int a_z = a.z;
	int b_z = b.z;

//	printf("dist: (%d,%d,%d) - (%d,%d,%d)\n", a_x, a_y, a_z, b_x, b_y, b_z);

	int d_x = min(abs(b_x - a_x), abs(b_x - (a_x + xyzsize.x)));
	int d_y = min(abs(b_y - a_y), abs(b_y - (a_y + xyzsize.y)));
	int d_z = min(abs(b_z - a_z), abs(b_z - (a_z + xyzsize.z)));

	int d=  d_x + d_y + d_z; 
	return d;
}


int main(int argc, char *argv[])
{
	FILE *fp;
	char line[80];
	int i, j, n;

	fp = fopen("topo.txt", "r");
	fgets(line, 80, fp);
	sscanf(line, "%d %d %d", &xyzsize.x, &xyzsize.y, &xyzsize.z);

	i = 0;
	while (fgets(line, 80, fp)!= NULL)
	{
		sscanf(line, "%d %d %d %d", &xyz[i].r, &xyz[i].x, &xyz[i].y, &xyz[i].z);
		i++;
	}
	fclose(fp);
	n = i;

	printf("%d %d %d\n", xyzsize.x, xyzsize.y, xyzsize.z);
	for (i = 0; i < n; i++) 
	{
		printf("%d %d %d %d\n", xyz[i].r, xyz[i].x, xyz[i].y, xyz[i].z);
		
	}

//	printf("dist(0,5)= %d\n", dist(0,5));
//	printf("dist(5,0)= %d\n", dist(5,0));
//	printf("dist(3,8)= %d\n", dist(3,8));

	printf("\n------------------------------\n");
	for (i = 0; i < n; i++) {
		for (j = 0; j < n; j++) {
			printf("%3d", dist(i,j));
		}
		printf("\n");
	}

	return 0;
}
