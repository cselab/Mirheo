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
int distances[MAX_NODES][MAX_NODES];

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

//i=63
//j=52
//dist: (2,0,2) - (1,0,7)
//dist(63,52)= 4
//i=52
//j=63
//dist: (1,0,7) - (2,0,2)
//dist(52,63)= 6


//	printf("dist: (%d,%d,%d) - (%d,%d,%d)\n", a_x, a_y, a_z, b_x, b_y, b_z);

	int d_x1 = min(abs(a_x - b_x), abs(a_x - (b_x + xyzsize.x)));
	int d_x2 = min(abs(b_x - a_x), abs(b_x - (a_x + xyzsize.x)));
	int d_y1 = min(abs(a_y - b_y), abs(a_y - (b_y + xyzsize.y)));
	int d_y2 = min(abs(b_y - a_y), abs(b_y - (a_y + xyzsize.y)));
	int d_z1 = min(abs(a_z - b_z), abs(a_z - (b_z + xyzsize.z)));
	int d_z2 = min(abs(b_z - a_z), abs(b_z - (a_z + xyzsize.z)));

	int d=  min(d_x1,d_x2) + min(d_y1,d_y2) + min(d_z1,d_z2); 
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

#if 1
	printf("\n------------------------------\n");
//	for (i = 0; i < n; i++) {
//		for (j = 0; j < n; j++) {
	int istart = 0;
	int iend = n-1;
	if (argc == 3) 
	{
		istart = atoi(argv[1]);
		iend = atoi(argv[2]);
	}

	for (i = istart; i <= iend; i++) {
		for (j = istart; j <= iend; j++) {
			distances[i][j] = dist(i, j);
			printf("%2d", dist(i,j));
		}
		printf("\n");
	}

#if 1
	int maxd = 0;
	for (i = istart; i <= iend; i++) {
		for (j = istart; j <= iend; j++) {
			if (distances[i][j] > maxd) maxd = distances[i][j];
		}
	}

	printf("maxd = %d\n", maxd);
	for (i = istart; i <= iend; i++) {
		for (j = i; j <= iend; j++) {
			if (distances[i][j] == maxd) 
				printf("%d-%d\n", i, j);
		}
	}	
#endif


#else
	while (1)
	{
		printf("i=");
		scanf("%d", &i);
		printf("j=");
		scanf("%d", &j);
		printf("dist(%d,%d)= %d\n", i, j, dist(i,j));
	}
#endif
	return 0;
}
