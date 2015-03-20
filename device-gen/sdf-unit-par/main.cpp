#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <iostream>
#include <vector>
using namespace std;

struct Parabola 
{
    float x0;
    float ymax; // length of obstacle, for cutting the pick
    float y0;

    Parabola(float gap, float xextent) : ymax(48.0) 
    {
        x0 = xextent/2.0f - gap/2.0f;
        if (gap > 9.0)
            y0 = ymax;
        else
            y0 = ymax + 6.5;
    }    

    void line1(vector<float>& vx, vector<float>& vy) {
        int N = 500;
        float dx = 2.0f * fabs(x0) / (N - 1);
        for (int i = 0; i < N; ++i) {
            float x = i * dx - x0;
            float y = 0.0f;
            vx.push_back(x);
            vy.push_back(y);
        }
    }

    void line2(vector<float>& vx, vector<float>& vy) {
        int N = 1500;
        float dx = 2.0f * fabs(x0) / (N - 1);
        float alpha = -y0 / (x0 * x0);
        for (int i = 0; i < N; ++i) {    
            float x = i * dx - x0;
            float y = min(ymax, alpha * x*x + y0);
            vx.push_back(x);
            vy.push_back(y);
        }
    }
};

int main(int argc, char ** argv)
{
    if (argc != 7)
    {
        printf("usage: ./sdf-unit <NX> <NY> <xextent> <yextent> <gap> <out-file-name> \n");
        return 1;
    }
    
    const int NX = atoi(argv[1]);
    const int NY = atoi(argv[2]);
    const float xextent = atof(argv[3]);
    const float yextent = atof(argv[4]);
    const float gap = atof(argv[5]);
 
    vector<float> xs, ys;
    Parabola par(gap, xextent);
    par.line1(xs, ys);
    par.line2(xs, ys);   

    const float xlb = -xextent/2.0f;
    const float ylb = -(yextent - par.ymax)/2.0f; 
    printf("starting brute force sdf with %d x %d starting from %f %f to %f %f\n",
           NX, NY, xlb, ylb, xlb + xextent, ylb + yextent);
    
    float * sdf = new float[NX * NY];
    const float dx = xextent / NX;
    const float dy = yextent / NY;
    const int nsamples = xs.size();
    
    for(int iy = 0; iy < NY; ++iy)
    for(int ix = 0; ix < NX; ++ix)
    {
        const float x = xlb + ix * dx;
        const float y = ylb + iy * dy;
        
        float distance2 = 1e6;
        int iclosest = 0;
        for(int i = 0; i < nsamples ; ++i)
        {
            const float xd = xs[i] - x;
            const float yd = ys[i] - y;
            const float candidate = xd * xd + yd * yd;
            
            if (candidate < distance2)
            {
                iclosest = i;
                distance2 = candidate;
            }
        }
    
        float s = -1;
    
        {
            const float alpha = -par.y0 / (par.x0 * par.x0);
            const float ycurve = min(par.ymax, alpha * x*x + par.y0);
            
            if (x >= -par.x0 && x <= par.x0 && y >= 0 && y <= ycurve)
                s = +1;
        }

        
        sdf[ix + NX * iy] = s * sqrt(distance2);
    }
    
    FILE * f = fopen(argv[6], "w");
    fprintf(f, "%f %f %f\n", xextent, yextent, 1.0f);
    fprintf(f, "%d %d %d\n", NX, NY, 1);
    fwrite(sdf, sizeof(float), NX * NY, f);
    fclose(f);
    
    delete [] sdf;
    
    return 0;
}
