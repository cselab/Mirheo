#include <iostream>
#include <cmath>
#include <cstring>
#include <cassert>
#include <fstream>
#include <stack>
#include <vector>
#include <algorithm>
#include <iomanip>
#include <unistd.h>
//#include <glew.h>

#ifdef __APPLE__
#include <GL/freeglut.h>
#include <OpenGL/OpenGL.h>
#else
#include <GL/glut.h>
#include <GL/gl.h>
#endif

#include "SmartCamera.h"
#include "GridWireframe.h"
//#include "screenshot.h"

int desired_frames = 240;

	
inline void LaunchError(const char * msg)
{
    //	MessageBox(NULL, msg,"ERROR", MB_OK);
    printf("\n->%s",msg);
    abort();
}
	
	
inline void CheckErrors(const char * sFile, int line)
{
    GLenum error = glGetError();
    if (error != GL_NO_ERROR)
    {
	printf("OpenGL ERROR: %d, FILE %s   LINE %d",(int)error,sFile, line);
	switch(error)
	{
	case GL_NO_ERROR:LaunchError("\n No error has been recorded  "); break;
	case GL_INVALID_ENUM:LaunchError("\n An unacceptable value is specified for an enumerated argument"); break; 
	case GL_INVALID_VALUE:LaunchError("\n A numeric argument is out of range."); break;
	case GL_INVALID_OPERATION: LaunchError("\nThe specified operation is not allowed in the current state."); break; 
	case GL_STACK_OVERFLOW: LaunchError("\nThis function would cause a stack overflow. "); break;
	case GL_STACK_UNDERFLOW: LaunchError("\nThis function would cause a stack underflow."); break; 
	case GL_OUT_OF_MEMORY: LaunchError("\nGL_OUT_OF_MEMORY."); break; 
	default: LaunchError("\nTHERE WAS AN ERROR, BUT NOT RECOGNIZED!!!"); break;
	}
    }
}
static inline void CheckOpenGLError(const char* stmt, const char* fname, int line)
{
    GLenum err = glGetError();
    if (err != GL_NO_ERROR)
    {
        printf("OpenGL error %08x, at %s:%i - for %s\n", err, fname, line, stmt);
        abort();
    }
}

#ifndef NDEBUG
#define GL_CHECK(stmt) do {				\
	stmt;						\
	CheckOpenGLError(#stmt, __FILE__, __LINE__);	\
    } while (0)
#else
#define GL_CHECK(stmt) stmt
#endif

std::vector<double> nice(std::vector<double> angles)
{
    std::vector<double> retval;
	
    const double x0 = angles.front();//*std::min_element(angles.begin(), angles.end());
	
    for(int i=0; i<angles.size(); ++i)
    {
	//	const double x0 = angles[i-1];
	const double x = angles[i];
	const double k = floor( 0.4999999999 + (x0 - x) / (2 * M_PI) );
		
	retval.push_back(x + k * 2 * M_PI);
    }
	
    //return angles;
    return retval;
}


class IW4 { 	
    static const int level = 10;
	
    static const int sH = -3;
    static const int eH = 4;
    static double Hs[7] ;//= {-1/16.,0,9/16.,1,9/16.,0,-1/16.};
	
    static const int nsize = 1 + ((1<<level)-1)*(eH-1- sH);
    static double data[nsize];
    static double s,e,h;
	
    static bool bInitialized;
	
	
    void _getPhysicalInfo(double& start, double& end, double& h)
	{
	    const int first = (pow((double)2, level)-1)*(1-eH);
		
	    h = pow(0.5, level);
	    start = first*h;
	    end = start + (nsize)*h;
		
	    assert(end == (-(pow(2.,1.*level)-1)*sH+1)*h);
	}
	
    void _generate(int l, int m, double w)
	{		
	    if (l == level)
	    {
		const int dest = m-(pow((double)2, level)-1)*(1-eH);
			
		assert(dest>=0);
		assert(dest<nsize);
			
		data[dest] += w;
	    }
	    else
	    {
		const int start = 2*m - eH + 1;
		const int end =  2*m - sH + 1;
			
		for(int k=start; k<end; k++)
		{
		    assert(2*m-k- sH>=0);
		    assert(2*m-k<eH);
				
		    _generate(l+1, k, w*Hs[2*m-k- sH]);
		}
	    }
	}
	
public:
    IW4()
	{
	    //printf("initialized is %d\n", bInitialized);
		
	    if (!bInitialized)
	    {
		bool bLUTFileExists = false;
			
		char buf[400];
		sprintf(buf, "IW4_LUT_level%d.serialized", level);
				

		{
		    FILE* f = fopen(buf, "rb");
				
		    bLUTFileExists = f!=NULL;
		    printf("bLUTFileExists %d\n", bLUTFileExists);	
		    if (bLUTFileExists)
		    {
			printf("Reading IW4 LUT...\n");
			fread((double *)data, sizeof(double), nsize, f);
			fclose(f);
		    }
				
				
				
		}
			
		if (!bLUTFileExists)
		{
		    for(int i=0; i<nsize; i++) data[i] = 0;
				
		    printf("Generating IW4 LUT...\n");
		    _generate(0, 0, 1);
				
		    FILE * f =  fopen(buf, "wb");
		    fwrite((double *)data, sizeof(double), nsize, f);
		    assert(f!=NULL);
		    fclose(f);
				
		}
			
		_getPhysicalInfo(s,e ,h);
			
		bInitialized = true;
	    }
	}
	
    static double eval(double x)
	{
	    if (x<s) return 0;
	    if (x>=e) return 0;
		
	    const double sampling_pos = (x-s)/h;
	    const double anchor_point = floor(sampling_pos);
	    const double r = sampling_pos - anchor_point;
	    const int ibase = (int)(anchor_point);
		
	    assert(ibase>=0 && ibase<nsize);
		
	    return (1-r)*data[ibase] + r*data[ibase+1];
	}
	
    void computeIthMoment(int ith)
	{
	    double m = 0;
		
	    for(int ix=0; ix<nsize; ix++)
	    {
		const double x = s + ix*h;
		m +=data[ix]*pow(x, ith);
	    }
		
	    m *= h;
		
	    printf("the moment m=%d is %f\n", ith, m);
	}
};

double IW4::data[nsize]; 
double IW4::s,IW4::e,IW4::h;
bool IW4::bInitialized = false;
double IW4::Hs[7] = {-1/16.,0,9/16.,1,9/16.,0,-1/16.};


double BS4(double x) 
{
    const double t = fabs(x);
	
    if (t>2) return 0;
	
    if (t>1) return pow(2-t,3)/6;
	
    return (1 + 3*(1-t)*(1 + (1-t)*(1 - (1-t))))/6;
}


double i4u(const double t,
	   const double f0, const double f1, const double f2, const double f3)
{
    /*const double a0 = f0;
      const double a1 = -11./6 * f0 + 3 * f1 - 3./2 * f2 + 1./3 * f3;
      const double a2 = f0 - 5./2 * f1 + 2 * f2 - 1./2 * f3;
      const double a3 = -1./6 * f0 + 1./2 *f1 - 1./2 * f2 + 1./6 * f3;
	 
      return a0 + t * (a1 + t * (a2 + t * a3));*/
	
    //return f0 * BS4(t - 0) + f1 * BS4(t - 1) + f2 * BS4(t - 2) + f3 * BS4(t - 3);
    IW4 iw4;
    return f0 * iw4.eval(t - 0) + f1 * iw4.eval(t - 1) + f2 * iw4.eval(t - 2) + f3 * iw4.eval(t - 3);

}

double i4nu(const double t,
	    const double f0, const double f1, const double f2, const double f3,
	    const double t0, const double t1, const double t2, const double t3)
{
    assert(t0 < t1);
    assert(t1 < t2);
    assert(t2 < t3);
	
    assert(t >= t1);
    assert(t <= t2);
	
    const double tau = 1 + (t - t1) / (t2 - t1);
	
    const double retval = i4u(tau, f0, f1, f2, f3);
	
    assert(!std::isnan(retval));

    return retval;
}

double i4nu_tvd(const double t,
		const double f0, const double f1, const double f2, const double f3,
		const double t0, const double t1, const double t2, const double t3)
{
    assert(t0 < t1);
    assert(t1 < t2);
    assert(t2 < t3);
	
    assert(t >= t1);
    assert(t <= t2);
	
    const double tau = 1 + (t - t1) / (t2 - t1);
	
    const double retval = i4u(tau, f0, f1, f2, f3);
	
    assert(!std::isnan(retval));
	
    const double refval[] = {f0, f1, f2, f3 };	
    const double fmax = *std::max_element(refval, refval + 4);
    const double fmin = *std::min_element(refval, refval + 4);
    const double hw = (fmax - fmin) / 2;
    const double center = (fmax + fmin) / 2;
    const double ampl = 1.01;
	
    return  std::max(center - ampl * hw , std::min(center + ampl * hw, retval));
    return retval;
}

struct PointCloud
{
    int npts = 0;
    std::vector<float> position, color;
    
    PointCloud() { }
    
    void load_from_file(std::ifstream & file)
	{
	    std::vector<float> p, c;
	    file.read((char*)&npts, sizeof(npts));

	    p.resize(npts * 3);
	    file.read((char *)&p.front(), sizeof(float) * 3 * npts);
	    
	    c.resize(npts * 3);
	    file.read((char *)&c.front(), sizeof(float) * 3 * npts);
	    
	    for(int i = 0; i < npts; ++i)
	    {
		const float x = p[0 + 3 * i];
		const float y = p[1 + 3 * i];
		const float z = p[2 + 3 * i];

		if (fabs(x - 0.5) < 4 &&
		    fabs(y - 0.5) < 4 &&
		    fabs(z - 0.5) < 4 )
		{
		    position.push_back(x);
		    position.push_back(y);
		    position.push_back(z);
		    
		    color.push_back(c[0 + 3 * i]);
		    color.push_back(c[1 + 3 * i]);
		    color.push_back(c[2 + 3 * i]);
		}
	    }

	    npts = position.size()/ 3;
	}
	
    void display()
	{
	    glPushAttrib(GL_ENABLE_BIT);

	    glEnable(GL_DEPTH_TEST);
	    glDisable(GL_LIGHT0);

	    glPushMatrix();

	    glPointSize(2);
	    glEnableClientState(GL_COLOR_ARRAY);	    
	    glEnableClientState(GL_VERTEX_ARRAY);
	    glVertexPointer(3, GL_FLOAT, 0, &position.front());
	    glColorPointer(3, GL_FLOAT, 0, &color.front());
	    GL_CHECK(glDrawArrays(GL_POINTS, 0, npts));
	    
	    glPopMatrix();
	    glPopAttrib();
	}
};

PointCloud pointcloud;

struct Path
{
    std::vector<double> xps, yps, zps, azimuths, elevations, timestamps;
    std::vector<bool> stopping;
	
    GLdouble sphereradius;

    void read_from_mvfile(const char * const path)
	{
	    printf("hello read_from_mvfile\n");
	    
	    std::ifstream input(path);
		
	    if(!input.good())
	    {
		std::cout << "Could not open the file. Aborting now.\n";
		abort();
	    }

	    int tstamp = 0;
	    
	    while (true)
	    {
		float MV[4][4];

		for(int i = 0; i < 16; ++i)
		    input >> MV[i / 4][i % 4];

		if (input.eof())
		    break;

		const double xp = -(MV[0][0] * MV[3][0] + MV[0][1] * MV[3][1] + MV[0][2] * MV[3][2]);
		const double yp = -(MV[1][0] * MV[3][0] + MV[1][1] * MV[3][1] + MV[1][2] * MV[3][2]);
		const double zp = -(MV[2][0] * MV[3][0] + MV[2][1] * MV[3][1] + MV[2][2] * MV[3][2]);

		const double xt = xp + -1 * MV[0][2];
		const double yt = yp + -1 * MV[1][2];
		const double zt = zp + -1 * MV[2][2];
				
		const double dx = xt - xp;
		const double dy = yt - yp;
		const double dz = zt - zp;
			
		//azimuth goes from 0 to 2 pi
		const double azimuth = atan2(dz, dx);
			
		//elevation goes from -pi/2 to pi/2
		const double elevation = atan2(dy, sqrt(dx * dx + dz * dz)); 
			
		std::cout << xps.size() << ") " << xp << " " <<  yp << " " <<  zp << " " <<
		    azimuth << " " <<  elevation << " " << tstamp << "\n";

		xps.push_back(xp);
		yps.push_back(yp);
		zps.push_back(zp);
		azimuths.push_back(azimuth);
		elevations.push_back(elevation);
		timestamps.push_back(tstamp);
		stopping.push_back(true);
		
		++tstamp;
	    }
	}

    void read_from_vpfile(const char * const path)
	{
	    std::ifstream input(path);
		
	    if(!input.good())
	    {
		std::cout << "Could not open the file. Aborting now.\n";
		abort();
	    }
		
	    while (input.good())
	    {
		double xp, yp, zp; //camera position
		double xt, yt, zt; //target position
		double tstamp;
		char sep;
			
		input >> xp >> yp >> zp >> sep >> xt >> yt >> zt >> sep >> tstamp;
			
		if (input.fail()) break;
			
		const double dx = xt - xp;
		const double dy = yt - yp;
		const double dz = zt - zp;
			
		//azimuth goes from 0 to 2 pi
		const double azimuth = atan2(dz, dx);
			
		//elevation goes from -pi/2 to pi/2
		const double elevation = atan2(dy, sqrt(dx * dx + dz * dz)); 
			
		std::cout << xps.size() << ") " << xp << " " <<  yp << " " <<  zp << " " <<
		    azimuth << " " <<  elevation << " " << tstamp << "\n";

		xps.push_back(xp);
		yps.push_back(yp);
		zps.push_back(zp);
		azimuths.push_back(azimuth);
		elevations.push_back(elevation);
		timestamps.push_back(tstamp);
		stopping.push_back(true);
	    }
	}
    
    Path(const char * const path) 
	{
	    {
		const int nchars = strlen(path);
		const bool mvfile = path[nchars-3] == '.' && path[nchars-2] == 'm' && path[nchars-1] == 'v';

		if (mvfile)
		    read_from_mvfile(path);
		else
		    read_from_vpfile(path);
	    }
		
	    if (xps.size() == 0)
	    {
		std::cout << "Could not read any entry. Aborting now.\n";
		abort();
	    }
		
	    if (xps.size() >= 2)
	    {
		stopping.front() = true;
		stopping.back() = true;
	    }
		
	    azimuths = nice(azimuths);
		
	    //check that timestamps are valid
	    {
		double currval = timestamps.front();
			
		for(int i = 1; i < xps.size(); ++i)
		{
		    assert(currval < timestamps[i]);
				
		    if (currval >= timestamps[i])
		    {
			std::cout << "Error! invalid timestamp!Aborting.\n";
			abort();
		    }
				
		    currval = timestamps[i];
		}
	    }
		
	    tube_colors[0] = 1;
	    tube_colors[1] = 1;
	    tube_colors[2] = 1;
	    sphereradius = 0.004;
	}
	
    Path()
	{
	    tube_colors[0] = 1;
	    tube_colors[1] = 1;
	    tube_colors[2] = 1;
	    sphereradius = 0.0036*0.8;
	}
	
    std::vector<double> cheat(std::vector<double> x, const double eps=1e-5)
	{
	    const double head[] = {x.front() - 3*eps, x.front() - 2*eps, x.front() - eps};
	    const double tail[] = {x.back() + eps, x.back() + 2*eps };
		
	    x.insert(x.begin(), head, head + 3);
	    x.insert(x.end(), tail, tail + 2);
		
	    return x;
	}
	
    Path refine(std::vector<double> desired_timestamps = std::vector<double>())
	{
	    Path retval;
		
	    const size_t NSTEPS = std::max((size_t)desired_frames, desired_timestamps.size());
	    double deltat = timestamps.back()/(NSTEPS-1);
		
	    std::vector<double> ts = cheat(timestamps);
	    std::vector<double> xs = cheat(xps);
	    std::vector<double> ys = cheat(yps);
	    std::vector<double> zs = cheat(zps);
	    std::vector<double> as = nice(cheat(azimuths));

	    std::vector<double> es = cheat(elevations);
		
	    const int NSAMPLES = ts.size();
	    //printf("NSTEPS IS %d\n", NSTEPS);
		
	    for(int istep = 0; istep < NSTEPS; ++istep)
	    {
		const double tdesired = (desired_timestamps.size() < NSTEPS) ? istep*deltat : desired_timestamps[istep]; 
		const int _base = -1 + std::lower_bound(ts.begin(), ts.end(), tdesired) - ts.begin();
			
		assert(_base > 0);
		assert(_base + 2 < NSAMPLES);
			
		const double tval = tdesired;//i4nu(istep*deltat, ts[_base-1], ts[_base], ts[_base+1], ts[_base+2], ts[_base-1], ts[_base], ts[_base+1], ts[_base+2]);
		const int base = -1 + std::lower_bound(ts.begin(), ts.end(), tval) - ts.begin();
			
		const double xval = i4nu_tvd(tval, xs[base-1], xs[base], xs[base+1], xs[base+2], ts[base-1], ts[base], ts[base+1], ts[base+2]);
		const double yval = i4nu_tvd(tval, ys[base-1], ys[base], ys[base+1], ys[base+2], ts[base-1], ts[base], ts[base+1], ts[base+2]);
		const double zval = i4nu_tvd(tval, zs[base-1], zs[base], zs[base+1], zs[base+2], ts[base-1], ts[base], ts[base+1], ts[base+2]);
			
		std::vector<double> azs;
		azs.push_back(as[base-1]);
		azs.push_back(as[base]);
		azs.push_back(as[base+1]);
		azs.push_back(as[base+2]);
				
		azs = nice(azs);
		const double azi = i4nu(tval, azs[0], azs[1], azs[2], azs[3], ts[base-1], ts[base], ts[base+1], ts[base+2]);
		const double ele = i4nu(tval, es[base-1], es[base], es[base+1], es[base+2], ts[base-1], ts[base], ts[base+1], ts[base+2]);
			
		retval.xps.push_back(xval);
		retval.yps.push_back(yval);
		retval.zps.push_back(zval);
			
		retval.azimuths.push_back(azi);
		retval.elevations.push_back(ele);			
			
		retval.timestamps.push_back(tdesired);
		retval.stopping.push_back(false);
	    }
		
	    retval.azimuths = nice(retval.azimuths);
		
	    return retval;
	}
	
    void _length_vs_time(const int NFINE, std::vector<double>& tsamples, std::vector<double>& lsamples)
	{
	    //printf("REFINE!!!\n");
	    if (xps.size() == 1)
	    {
		printf("hey this is the case\n");
		return ;//*this;
	    }
		

	    Path refined;
		
	    {
		const double dtinsane = timestamps.back() / (NFINE - 1);
		std::vector<double > tinsane;
			
		for (int i=0; i<NFINE; ++i)
		    tinsane.push_back(i*dtinsane);
			
		refined =refine(tinsane);
	    }
		
	    //compute total length, fill up a vector of < ti, li>
	    double totlength = 0;
	    tsamples.clear();
	    lsamples.clear();	
	    for (int i=0; i<refined.xps.size()-1; ++i)
	    {
		tsamples.push_back(refined.timestamps[i]);
		lsamples.push_back(totlength);
			
		const double r = 5e-1;

		double x0, y0, z0;
		{
		    const double a = refined.azimuths[i];
		    const double e = refined.elevations[i];
				
		    x0 = refined.xps[i] + r * cos(a) * cos(e);
		    y0 = refined.yps[i] + r * sin(e);
		    z0 = refined.zps[i] + r * sin(a) * cos(e);
		}
			
		double x1, y1, z1;
		{
		    const double a = refined.azimuths[i+1];
		    const double e = refined.elevations[i+1];
				
		    x1 = refined.xps[i+1] + r * cos(a) * cos(e);
		    y1 = refined.yps[i+1] + r * sin(e);
		    z1 = refined.zps[i+1] + r * sin(a) * cos(e);
		}
			
		totlength += sqrt(pow(x1 - x0, 2) + 
				  pow(y1 - y0, 2) + 
				  pow(z1 - z0, 2));
		/*
		  totlength += sqrt(pow(refined.xps[i+1] - refined.xps[i], 2) + 
		  pow(refined.yps[i+1] - refined.yps[i], 2) + 
		  pow(refined.zps[i+1] - refined.zps[i], 2));			*/
			
	    }

	    tsamples.push_back(timestamps.back());
	    lsamples.push_back(totlength);
	}
	
    Path refine_uniform(const int NFINE)
	{
	    struct { int levels, start, end; } stopinfo = { 6, -3, 3};

	    const int BFIRST = std::count(stopping.begin(), stopping.begin() + 1, true);
	    const int B = stopping.size() <= 2 ? 0 : std::count(stopping.begin() + 1, stopping.end() - 1, true);
	    const int BLAST = std::count(stopping.end() - 1, stopping.end(), true);
		
	    const int M = stopinfo.levels * (stopinfo.end - stopinfo.start);
		
	    const int C = std::max(2, desired_frames - M/2 * (BFIRST + BLAST) - M * B);
		
	    printf("B1 %d B=%d BL = %d M=%d C=%d\n", BFIRST, B, BLAST, M, C);
		
	    //early exit criteria
	    {
		if (xps.size() == 1)
		    return *this;
			
		double bbox[2][3];
		boundingbox(bbox);
			
		const double charlen = std::max( std::max(bbox[1][2]-bbox[0][2], bbox[1][1]-bbox[0][1]), bbox[1][0]-bbox[0][0]);
			
		if (charlen < 1e-4)
		    return refine();
	    }
		
	    std::vector< double > tsamples, lsamples;
	    _length_vs_time(NFINE, tsamples, lsamples);
	
	    const double totlength = lsamples.back();
	    const double ds_cruise = totlength/(C-1);
		
	    //put all the sample locations here
	    std::vector< double > desired_s;
		
	    //put cruise points
	    for(int is = 1; is < C-1; ++is)
		desired_s.push_back(is * ds_cruise);
		
	    //put stopping points
	    for(int i=0; i<stopping.size(); ++i)
	    {
		if (!stopping[i]) continue;
			
		const int tentry = std::lower_bound(tsamples.begin(), tsamples.end(), timestamps[i]) - tsamples.begin();
			
		assert(tentry < tsamples.size());
		assert(tentry >= 0);

		const double myl = lsamples[tentry];
			
		for(int j = 0; j < stopinfo.levels; ++j)
		{
		    const double ds = ds_cruise * pow(0.5, j);
				
		    const double lanchor = ds * floor(myl / ds);
				
		    for(int k = stopinfo.start; k < stopinfo.end; ++k)
		    {
			const double planned_s = lanchor + (k + 0.5) * ds;
					
			if (planned_s > 0 && planned_s < totlength)
			    desired_s.push_back(planned_s);
		    }
		}
	    }
		
	    std::sort(desired_s.begin(), desired_s.end());
		
	    //the arclength support of the stop was too big
	    if (desired_s.size() != desired_frames - 2)
	    {
		std::cout << "Warning: i could not keep the number of frames to " << desired_frames <<
		    ", now the number of frames are " << desired_s.size() + 2 << std::endl;
	    }
				
	    //find time stamps of the samples
	    std::vector< double > finalstamps(1, 0);
	    for(std::vector<double>::const_iterator itS = desired_s.begin(); itS != desired_s.end(); ++itS)
	    {			
		const int entry = std::lower_bound(lsamples.begin(), lsamples.end(), *itS) - lsamples.begin() - 1;
			
		assert(entry >= 0);
		assert(entry + 1 < lsamples.size());
			
		const double tfinal = tsamples[entry] + (*itS - lsamples[entry]) * (tsamples[entry + 1] - tsamples[entry]);
			
		finalstamps.push_back(tfinal);
	    }
	    finalstamps.push_back(tsamples.back());

	    Path retval = refine(finalstamps);

	    return retval;
	}
	
    Path refine_uniform0(const int NFINE)
	{
	    //printf("REFINE %d!!!\n", NFINE);
	    if (xps.size() == 1)
		return *this;
		
	    std::vector< double > tsamples, lsamples;
		
	    _length_vs_time(NFINE, tsamples, lsamples);
		
	    const double totlength = lsamples.back();
		
	    if (totlength < 1e-1)
	    {
		printf("singularity in the total length!\n");
		return refine();
	    }
	    else
	    {
		printf("length is ok : %f\n", totlength);
	    }
		
	    const int N = desired_frames;
	    const double ds = totlength/(N-1);
		
	    std::vector< double > finalstamps;
		
	    finalstamps.push_back(0);
	    for(int is = 1; is < N-1; ++is)
	    {			
		const int entry = std::lower_bound(lsamples.begin(), lsamples.end(), is * ds) - lsamples.begin() - 1;
			
		assert(entry >= 0);
		assert(entry + 1 < lsamples.size());
			
		const double tfinal = tsamples[entry] + (is * ds - lsamples[entry]) * (tsamples[entry + 1] - tsamples[entry]);
			
		finalstamps.push_back(tfinal);
	    }
	    finalstamps.push_back(tsamples.back());
		
	    Path retval = refine(finalstamps);
		
	    return retval;
	}
	
    std::vector<double> _smooth(std::vector<double> x)
	{
	    std::vector<double> retval;
		
	    retval.push_back(x.front());
		
	    for(int i=1; i<x.size()-1; ++i)
		retval.push_back(x[i-1]*0.25 + x[i]*0.5 + x[i+1]*0.25);
		
	    retval.push_back(x.back());
		
	    return retval;
	}
	
    std::vector<double> _smooth_nice(std::vector<double> x)
	{
	    std::vector<double> retval;
		
	    retval.push_back(x.front());
		
	    for(int i=1; i<x.size()-1; ++i)
	    {
		std::vector<double> xs;
		xs.push_back(x[i-1]);
		xs.push_back(x[i]);
		xs.push_back(x[i+1]);
		xs = nice(xs);
		//retval.push_back(x[i-1]*0.25 + x[i]*0.5 + x[i+1]*0.25);
		const double tmp = xs[0]*0.25 + xs[1]*0.5 + xs[2]*0.25;
		//const double tmp = xs[i-1]*0.25 + xs[i]*0.5 + xs[i+1]*0.25;
		xs.push_back(tmp);
		xs = nice(xs);
		retval.push_back(xs.back());
	    }
		
	    retval.push_back(x.back());
		
	    return retval;
	}
	
    void smooth(const int niters  = 10)
	{
	    for(int i=0; i<niters; ++i)
	    {
		xps = _smooth(xps);
		yps = _smooth(yps);
		zps = _smooth(zps);
		azimuths = nice(_smooth_nice(nice(azimuths)));
		elevations = _smooth(elevations);
	    }
	}
	
    int size() const {  return xps.size(); }
	
    void boundingbox(double bbox[2][3])
	{
	    bbox[0][0] = bbox[1][0] = xps.front();
		
	    for(int i=0; i<xps.size(); ++i)
	    {
		bbox[0][0] = std::min(bbox[0][0], xps[i]);
		bbox[1][0] = std::max(bbox[1][0], xps[i]);
	    }
		
	    bbox[0][1] = bbox[1][1] = yps.front();
		
	    for(int i=0; i<yps.size(); ++i)
	    {
		bbox[0][1] = std::min(bbox[0][1], yps[i]);
		bbox[1][1] = std::max(bbox[1][1], yps[i]);
	    }
		
	    bbox[0][2] = bbox[1][2] = zps.front();
		
	    for(int i=0; i<zps.size(); ++i)
	    {
		bbox[0][2] = std::min(bbox[0][2], zps[i]);
		bbox[1][2] = std::max(bbox[1][2], zps[i]);
	    }
	}
    //void CheckErrors(const char * s="", int line=0);

	
    int closest(const double x, const double y, const double z)
	{
	    printf("CLOSEST DATA %f %f %f\n", x, y, z);
	    std::vector<double> distances;
		
	    for(int i=0; i<xps.size();++i)
	    {
		const double dx = x - xps[i];
		const double dy = y - yps[i];
		const double dz = z - zps[i];
			
		distances.push_back( dx * dx + dy * dy + dz * dz );
	    }
		
	    return std::min_element(distances.begin(), distances.end()) - distances.begin();
	}
	
    void _gl_geometrytube(const int NA = 100, const int NZ = 10)
	{
	    const double dalpha = 2 * M_PI / (NA - 1);
	    const double dz = 1. / (NZ - 1);
		
	    for(int iz = 0; iz < NZ - 1; ++iz)
	    {
		glBegin(GL_TRIANGLE_STRIP);

		for (int ia = 0; ia<NA; ++ia) 
		    for(int d = 0; d < 2; ++d)
		    {
			glNormal3d(cos(ia * dalpha), sin(ia * dalpha), iz*dz);
			glVertex3f(cos(ia * dalpha), sin(ia * dalpha), (iz + d) * dz);
		    }
			
		glEnd();
	    }
	}
	
    void _gl_geometrybbox()
	{
	    double bbox[2][3];
	    boundingbox(bbox);
		
	    for(int iz=0; iz<2; ++iz)
	    {
		glBegin(GL_LINE_LOOP);
		glVertex3d(bbox[0][0], bbox[0][1], bbox[iz][2]);
		glVertex3d(bbox[1][0], bbox[0][1], bbox[iz][2]);
		glVertex3d(bbox[1][0], bbox[1][1], bbox[iz][2]);
		glVertex3d(bbox[0][0], bbox[1][1], bbox[iz][2]);		
		glEnd();
	    }
		
	    for(int ix=0; ix<2; ++ix)
	    {
		glBegin(GL_LINE_LOOP);
		glVertex3d(bbox[ix][0], bbox[0][1], bbox[1][2]);
		glVertex3d(bbox[ix][0], bbox[0][1], bbox[0][2]);
		glVertex3d(bbox[ix][0], bbox[1][1], bbox[0][2]);
		glVertex3d(bbox[ix][0], bbox[1][1], bbox[1][2]);		
		glEnd();
	    }
		
	    for(int iy=0; iy<2; ++iy)
	    {
		glBegin(GL_LINE_LOOP);
		glVertex3d(bbox[1][0], bbox[iy][1], bbox[1][2]);
		glVertex3d(bbox[1][0], bbox[iy][1], bbox[0][2]);
		glVertex3d(bbox[0][0], bbox[iy][1], bbox[0][2]);
		glVertex3d(bbox[0][0], bbox[iy][1], bbox[1][2]);		
		glEnd();
	    }
	}
	
    void _gl_statusmaterial()
	{
	    {
		/*GLfloat white[4] = {1,1,1,1};
			
		  GLfloat * mat_specular = white;
		  GLfloat mat_shinines = 50.0;
			
		  GLfloat lightpos[4] = {10,10,10,1};
		  //GLfloat * light_color = white;
			
		  GLfloat light_ambient[4]= {0.3, 0.3, 0.3, 1.};
			
		  glShadeModel(GL_SMOOTH);
		  glMaterialfv(GL_FRONT, GL_SPECULAR, mat_specular);
		  glMaterialf(GL_FRONT, GL_SHININESS, mat_shinines);
		*/
			
		glEnable(GL_LIGHTING);
		glEnable(GL_LIGHT0);
		glEnable(GL_NORMALIZE);
		glEnable(GL_DEPTH_TEST);
		glEnable(GL_COLOR_MATERIAL);
		glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE);
	    }
	}
	
    void paint(double r, double g, double b, const bool painttubes = true)
	{
	    const int N = xps.size();
		
	    std::vector<double> rs(N, r), gs(N, g), bs(N, b);
		
	    paint(rs, gs, bs, painttubes);
	}
	
    GLfloat tube_colors[3];
	
    void set_tube_colors(double r, double g, double b)
	{
	    tube_colors[0] = r;
	    tube_colors[1] = g;
	    tube_colors[2] = b;
	}
	
    void paint(std::vector<double> rs, std::vector<double> gs, std::vector<double> bs, const bool painttubes = true)
	{
	    glPushAttrib(GL_POINT_BIT | GL_CURRENT_BIT | GL_ENABLE_BIT);
				
	    //paint the path as white linestrip
	    glBegin(GL_LINE_STRIP);		
	    glColor3d(1, 1, 1);
	    for(int i=0; i<xps.size(); ++i)
		glVertex3d(xps[i], yps[i], zps[i]);
	    glEnd();
		
	    //paint the bounding box?
	    glColor3d(0, 1, 1);
	    _gl_geometrybbox();
		
	    //draw the path with solid tubes and spheres
	    _gl_statusmaterial();
				
	    for(int i = 0; i < xps.size(); ++i)
	    {			
		glPushMatrix();
			
		glTranslated(xps[i], yps[i], zps[i]);
			
		//GLfloat c3[] = {rs[i],gs[i],bs[i],1.};
		//glMaterialfv(GL_FRONT, GL_AMBIENT_AND_DIFFUSE, c3);
		glColor3f(rs[i],gs[i],bs[i]);
			
		const double r = 0.003;
			
		if (!stopping[i])
		    glutSolidSphere(sphereradius, 20, 20);
		else
		    glutSolidCube(sphereradius * sqrt(3.));
			
		//glRasterPos2i(100, 120);
		//glColor4f(0.0f, 0.0f, 1.0f, 1.0f);
		//unsigned char buf[200] =  "culo culo";
		//glutBitmapString(GLUT_BITMAP_HELVETICA_18, buf);
			
		if (!painttubes || i >= xps.size() - 1) 
		{
		    glPopMatrix();
		    continue;
		}
			
			
		const double dx = xps[i+1] - xps[i];
		const double dy = yps[i+1] - yps[i];
		const double dz = zps[i+1] - zps[i];
		const double length = sqrt(dx*dx + dy*dy + dz*dz);
		const double azimuth = atan2(dz, dx);
		const double elevation = atan2(dy, sqrt(dx*dx + dz*dz));
			
		glRotatef((-azimuth + M_PI / 2) * 180 / M_PI, 0, 1, 0);
		glRotatef(-elevation * 180 / M_PI, 1, 0, 0);
			
		glScalef( r, r, length);
			
		//const GLfloat white[] = {1,1,1,1.};
		//glMaterialfv(GL_FRONT, GL_AMBIENT_AND_DIFFUSE, white);
		//const double lambda = 0.25;
		//glColor3f(rs[i]*lambda + 1-lambda,gs[i]*lambda + 1-lambda ,bs[i]*lambda + 1-lambda);
		glColor3fv(tube_colors);
		_gl_geometrytube();
			
		glPopMatrix();
	    }
		
	    glPopAttrib();
	}
	
    void follow(int frame, SmartCamera& camera)
	{
	    camera.move_to(xps[frame], yps[frame], zps[frame]);
	    camera.point_to(azimuths[frame], elevations[frame]);
	}
	
    void save(SmartCamera& camera, const char * path = "camera-edited.txt")
	{
	    glPushMatrix();

	    {
		std::ofstream myout(path);
			
		for(int frame=0; frame<timestamps.size(); ++frame)
		{
		    camera.move_to(xps[frame], yps[frame], zps[frame]);
		    camera.point_to(azimuths[frame], elevations[frame]);
				
		    GLfloat MV[4*4];
				
		    glGetFloatv(GL_MODELVIEW_MATRIX, MV);
				
		    for(int c=0; c<15; ++c)
			myout << MV[c] << " ";
				
		    myout << MV[15] << "\n";
		}
	    }		
		
	    glPopMatrix();
	}
	
    void get(const int index, double& x, double& y, double& z)
	{
	    assert(index >= 0 && index < xps.size());
		
	    x = xps[index];
	    y = yps[index];
	    z = zps[index];
	}
	
    void get(std::vector<double>& x, std::vector<double>& y, std::vector<double>& z)
	{
	    x = xps;
	    y = yps;
	    z = zps;
	}
	
    void get(const int index, double& x, double& y, double& z, double& azimuth, double& elevation)
	{
	    assert(index >= 0 && index < xps.size());
		
	    x = xps[index];
	    y = yps[index];
	    z = zps[index];
	    azimuth = azimuths[index];
	    elevation = elevations[index];
	}
	
	
    void ptr(const int index, double *& x, double *& y, double *& z, double *& azimuth, double *& elevation)
	{
	    assert(index >= 0 && index < xps.size());
		
	    x = &xps[index];
	    y = &yps[index];
	    z = &zps[index];
	    azimuth = &azimuths[index];
	    elevation = &elevations[index];
	}
};

struct EditablePath: public Path
{
    int selected_ctrlpt;
	
    struct Redisplacement 
    { 
	double xnew, ynew, znew, xold, yold, zold; 
	int isel;
    }; 
	
    std::stack< Redisplacement > actiontrace;
	
    EditablePath(const char * const path): Path(path), selected_ctrlpt(-1){}
	
    EditablePath(): Path(), selected_ctrlpt(-1){}
	
    int select_closest(double x, double y, double z)
	{
	    return selected_ctrlpt = closest(x, y, z);
	}
	
    void select(int index)
	{
	    selected_ctrlpt = index;
	}
	
    void get_selected(double& x, double& y, double& z, double& a, double& e)
	{
	    if (selected_ctrlpt >= 0 && selected_ctrlpt < xps.size())
		Path::get(selected_ctrlpt, x, y, z, a, e);
	}
	
    void set_selected(double x, double y, double z, double a, double e)
	{
	    if (selected_ctrlpt >= 0 && selected_ctrlpt < xps.size())
	    {
		xps[selected_ctrlpt] = x;
		yps[selected_ctrlpt] = y;
		zps[selected_ctrlpt] = z;
		azimuths[selected_ctrlpt] = a;
		elevations[selected_ctrlpt] = e;
	    }
	}
	
    void stopping_flip() 
	{ 
	    if (selected_ctrlpt >= 0 && selected_ctrlpt < xps.size())
		stopping[selected_ctrlpt] = (bool)(1-(int)stopping[selected_ctrlpt]);
	}
	
    void paint(double r, double g, double b)
	{
	    const int N = xps.size();
		
	    std::vector<double> rs(N, r), gs(N, g), bs(N, b);
		
	    if (selected_ctrlpt >= 0 && selected_ctrlpt < N)
	    {
		//set the ctrl point as orange/yellow
		rs[selected_ctrlpt] = 255/255.;
		gs[selected_ctrlpt] = 176/255.;
		bs[selected_ctrlpt] = 67/255.;
	    }
	
	    Path::paint(rs, gs, bs);
	}
	
    void redisplace(double x, double y, double z)
	{
	    assert(selected_ctrlpt >= 0 && selected_ctrlpt < xps.size());
		
	    Redisplacement rd = { x, y, z, xps[selected_ctrlpt], yps[selected_ctrlpt], zps[selected_ctrlpt], selected_ctrlpt } ;
		
	    xps[selected_ctrlpt] = x;
	    yps[selected_ctrlpt] = y;
	    zps[selected_ctrlpt] = z;
		
	    actiontrace.push(rd);
	}
	
    void undo()
	{
	    if (actiontrace.size() == 0) return;
		
	    Redisplacement rd = actiontrace.top();
		
	    actiontrace.pop();
		
	    xps[rd.isel] = rd.xold;
	    yps[rd.isel] = rd.yold;
	    zps[rd.isel] = rd.zold;
		
	    printf("undoing now\n");
	}	
};

SmartCamera camera;

GridWireframe grid(-1,-0.5,-0.5, 2,1.5,2, 25);
EditablePath path;//("/Users/diegor/Camera_APS_FT_10000.dat");
Path referencepath;
Path refinedpath;


struct NavigationData
{
    bool refvalid;
    int xref, yref;
    float zref;
	
    NavigationData():refvalid(false)
	{
	}
};

struct EditData
{
    float measured_depth;
    int sel_ctrlpoint;
    bool selected;
};

NavigationData navigationdata;
EditData editdata;
SmartCamera::Pos testpos0 = {0,0,0};
SmartCamera::Pos testpos1 = {0,0,0};

struct ArrowGeometry
{
    const double L, l, b, H, h;
	
    ArrowGeometry(): L(10), l(4), b(3), H(4), h(1.8) {}
	
    void paint2d()
	{
	    glBegin(GL_POLYGON);
	    glColor3d(1, 1, 0);
	    glVertex2d(0, 0);
	    glVertex2d(-b, H/2);
	    glVertex2d(-b, h/2);
	    glVertex2d(-l, h/2);
	    glVertex2d(-l, -h/2);
	    glVertex2d(-b, -h/2);
	    glVertex2d(-b, -H/2);
	    glVertex2d(0, 0);
	    glEnd();
		
	    glPushAttrib(GL_ENABLE_BIT);
		
	    glEnable(GL_BLEND);
	    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	    glBegin(GL_QUADS);
	    glColor4d(1, 0.2, 0, 0);
	    glVertex2d(-L,-h/2);
	    glColor4d(1, 1, 0, 1);
	    glVertex2d(-l,-h/2);
	    glVertex2d(-l, h/2);
	    glColor4d(1, 0.2, 0, 0);
	    glVertex2d(-L, h/2);
	    glEnd();
		
	    glPopAttrib();
	}
	
    void paint3d(const double red = 0.5, const double green = 0.5, const double blue = 0.5, const double alpha = 0.5)
	{
	    const int NA = 30;
	    const double da = 2 * M_PI / (NA - 1);
	
	    glPushAttrib(GL_CURRENT_BIT | GL_ENABLE_BIT);

	    GLfloat white[4] = {1,1,1,1};
	    GLfloat color[4] = {red, green, blue, alpha};
	    glMaterialfv(GL_FRONT, GL_DIFFUSE, white);
	    glMaterialfv(GL_FRONT, GL_AMBIENT, color);
		
	    glEnable(GL_LIGHT0);
	    glEnable(GL_LIGHTING);
	    glEnable(GL_NORMALIZE);
		
	    //prepuzio
	    glBegin(GL_TRIANGLE_FAN);
	    glNormal3d(1, 0, 0);
	    glVertex3d(0, 0, 0);
	    for(int ia = 0; ia < NA; ++ia)
	    {
		const double a = ia * da;
		const double beta = M_PI - atan2(h/2, b) - M_PI/2;
			
		glNormal3d(cos(beta), sin(beta)*cos(a), sin(beta)*sin(a));
		glVertex3d(-b, H/2*cos(a), H/2*sin(a));
	    }
	    glEnd();
		
	    //tappo
	    glBegin(GL_TRIANGLE_FAN);
	    glNormal3d(-1, 0, 0);
	    glVertex3d(-b, 0, 0);
	    for(int ia = 0; ia < NA; ++ia)
	    {
		const double a = ia * da;
		glNormal3d(-1, 0, 0);
		glVertex3d(-b, H/2*cos(a), H/2*sin(a));
	    }
	    glEnd();		
		
	    //uccello
	    glBegin(GL_TRIANGLE_STRIP);
	    for(int ia = 0; ia < NA; ++ia)
	    {
		const double a = ia * da;
			
		for(int iz=0; iz<2; ++iz)
		{
		    glNormal3d(0, cos(a), sin(a));
		    glVertex3d(-L*(1-iz) + -b*iz, h/2*cos(a), h/2*sin(a));
		}
	    }
	    glEnd();
		
	    //tappo
	    glBegin(GL_TRIANGLE_FAN);
	    glNormal3d(-1, 0, 0);
	    glVertex3d(-L, 0, 0);
	    for(int ia = 0; ia < NA; ++ia)
	    {
		const double a = ia * da;
			
		glNormal3d(-1, 0, 0);
		glVertex3d(-L, h/2*cos(a), h/2*sin(a));
	    }
	    glEnd();
				
	    glPopAttrib();
	}
	
    void paintframe()
	{
	    {
		glPushMatrix();		
		glTranslatef(L, 0, 0);
		paint3d(1,0,0,1);
		glPopMatrix();
	    }
		
	    {
		glPushMatrix();
		glRotatef(-90, 0, 1, 0);
		glTranslatef(L, 0, 0);
		paint3d(0,0,1,1);
		glPopMatrix();
	    }
		
		
	    {
		glPushMatrix();
		glRotatef(-90, 0, 0, -1);
		glTranslatef(L, 0, 0);
		paint3d(0,1,0,1);
		glPopMatrix();
	    }
	}		
};

struct TrackballAxis
{
    bool valid;
	
    double xpixelcenter, ypixelcenter;
    double pixelradius;//, depth;
	
    double xref, yref, zref;
	
    void setref(double x, double y, double z)
	{
	    xref = x;
	    yref = y;
	    zref = z;
	}
	
    void update(double x, double y, double z, double& azi, double& ele)
	{
	    const double aziref = atan2(zref , xref );
	    const double eleref = atan2(yref , sqrt(pow(xref ,2) + pow(zref ,2)));
		
	    const double azinew = atan2(z , x );
	    const double elenew = atan2(y , sqrt(pow(x,2) + pow(z ,2)));
		
	    azi = azinew - aziref;
	    ele = elenew - eleref;
	}
} trackballaxis;

struct PlayData
{
    bool valid;
    int frame;
	
} playdata;

void display()
{
    GLdouble VP[4];
    glGetDoublev(GL_VIEWPORT, VP);
	
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);

    {
	glPushMatrix();
	glRotatef(-90, 1, 0, 0);
	glBegin(GL_LINES);
	
	glVertex2f(0, 0);
	glVertex2f(1,1);
	glVertex2f(0,1);
	glVertex2f(1, 0);
	glEnd();
	glPopMatrix();
    }
    
    //display caxes
    {
	const double k = 0.01;
	glPushMatrix();
	glScalef(k, k, k);
	ArrowGeometry arrow;
	arrow.paintframe();
	glPopMatrix();
    }
	
    grid.paint(0.25);
    refinedpath.paint(1,0.1,0.1, false);
	
    pointcloud.display();
	
    if (playdata.valid)
    {
	glutSwapBuffers();
	return;
    }
	
    path.paint(1,1,1);
    referencepath.paint(0.6, 0.6, 0.8);

    //draw the start arrow
    {
	double angle, xdisplacement, ydisplacement;
		
	{
	    std::vector<double> xs, ys, zs;
	    path.get(xs, ys, zs);
	    std::vector<double> pxs, pys, pzs;
			
	    camera.points2pixels(xs, ys, zs, pxs, pys, pzs);
			
	    double xcenter = 0, ycenter = 0;
	    int counter = 0;
	    for(int i= 0; i < pxs.size(); ++i)
	    {
		if (pxs[i] < 0 || pxs[i] >= VP[2]) continue;
		if (pys[i] < 0 || pys[i] >= VP[3]) continue;
				
		xcenter += pxs[i];
		ycenter += pys[i];
		counter ++;
	    }
			
	    xcenter /= counter;
	    ycenter /= counter;
			
	    if (counter <= 1)
	    {
		xcenter = 0.5 * (pxs[0] + pxs[1]);
		ycenter = 0.5 * (pys[0] + pys[1]);
	    }
			
	    const double _xdirection = pxs[0] - xcenter;
	    const double _ydirection = pys[0] - ycenter;
	    const double f = 1./sqrt(pow(_xdirection,2) + pow(_ydirection, 2));
	    const double xdir = f * _xdirection;
	    const double ydir = f * _ydirection;
			
	    angle = atan2(ydir, xdir);
	    xdisplacement = pxs[0];
	    ydisplacement = pys[0];
	}
		
	glPushMatrix();
	glLoadIdentity();
	glMatrixMode(GL_PROJECTION);
	glPushMatrix();
	glLoadIdentity();
	glOrtho(VP[0], VP[0] + VP[2],  VP[1]+ VP[3] ,VP[1], -1, 1);
	glMatrixMode(GL_MODELVIEW);
		
	ArrowGeometry arrow;
	glColor3f(1, 1, 1);
	glTranslated(xdisplacement, ydisplacement, 0);
	glRotatef(angle * 180 / M_PI + 180, 0, 0, 1);
	glTranslated(-20, 0, 0);
	const double scaling = 10;
	glScalef(scaling, scaling, scaling);
		
	arrow.paint2d();
		
	glPopMatrix();
	glMatrixMode(GL_PROJECTION);
	glPopMatrix();
	glMatrixMode(GL_MODELVIEW);
    }
	
    //paint orientation-editing subwindow
    if (trackballaxis.valid)
    {
	{
	    const double k = 2e-3;
	    double x,y,z, a,e;
	    path.get_selected(x,y,z, a,e);

	    ArrowGeometry arrow;
			
	    //paint the orientation system at the control point
	    glPushMatrix();
	    glTranslated(x, y, z);
	    glRotatef(a * 180 / M_PI + 90, 0, -1, 0);
	    glRotatef(e * 180 / M_PI, 1, 0, 0);
	    glScalef(k, k, k);
	    arrow.paintframe();
			
	    {
		glPushMatrix();
		glRotatef(90, 0, 1, 0);
		glTranslatef(arrow.L, 0, 0);
		arrow.paint3d(0.75,0.75,0.75,1);
		glPopMatrix();
	    }
			
	    glPopMatrix();
					
	    //find the right displacement of the orientation-editing subwindow
	    SmartCamera::Pos p = camera.project2pixels(x,y,z);
			
	    const double pixel_distance = 90;
	    const double angle = 0.75*M_PI;
	    const double xcenter = p.x + pixel_distance * cos(angle);
	    const double ycenter = p.y - pixel_distance * sin(angle);
						
	    trackballaxis.xpixelcenter = xcenter;
	    trackballaxis.ypixelcenter = ycenter;
	    trackballaxis.pixelradius = 60;
			
	    //draw the circular subwindow
	    {
		glPushAttrib(GL_ENABLE_BIT | GL_CURRENT_BIT);
		glDisable(GL_DEPTH_TEST);
		glDisable(GL_LIGHTING);
				
		glPushMatrix();
		glLoadIdentity();
		glMatrixMode(GL_PROJECTION);
		glPushMatrix();
		glLoadIdentity();
				
		glOrtho(VP[0], VP[0] + VP[2],  VP[1]+ VP[3] ,VP[1], 0, 1);
		glMatrixMode(GL_MODELVIEW);
				
		glTranslated(xcenter, ycenter, 0);
				
		glScalef(trackballaxis.pixelradius, trackballaxis.pixelradius, trackballaxis.pixelradius);
		glColor3d(1, 1, 1);
				
		const int NA = 100;
		const double da = 2 * M_PI / (NA - 1);
				
		glBegin(GL_TRIANGLE_FAN);
		glColor3d(0, 0, 0);
		glVertex2d(0,0);
		for (int i=0; i< NA; ++i)
		    glVertex2d(cos(i*da), sin(i*da));
		glEnd();
				
		glBegin(GL_LINE_LOOP);
		glColor3d(1, 1, 1);
		for (int i=0; i< NA; ++i)
		    glVertex2d(cos(i*da), sin(i*da));
		glEnd();
				
				
		glPopMatrix();
		glMatrixMode(GL_PROJECTION);
		glPopMatrix();
		glMatrixMode(GL_MODELVIEW);
				
		glPopAttrib();
	    }
			
	    //paint the system of reference in the subwindow
	    {
		glPushAttrib(GL_VIEWPORT_BIT);
				
		const double h = 2 * trackballaxis.pixelradius;
		const double w = VP[2]/VP[3] * h;
		glViewport(trackballaxis.xpixelcenter - w/2, VP[3] - trackballaxis.ypixelcenter - h/2, w, h);
				
		glClear(GL_DEPTH_BUFFER_BIT);

		glPushMatrix();
		glRotatef(a * 180 / M_PI + 90, 0, -1, 0);
		glRotatef(e * 180 / M_PI, cos(a + M_PI/2), 0, sin(a+ M_PI/2));
				
		camera.zadjust(0, 0, 0, 12*sqrt(3));
				
		arrow.paintframe();

		{
		    glPushMatrix();
		    glRotatef(90, 0, 1, 0);
		    glTranslatef(arrow.L, 0, 0);
		    arrow.paint3d(0.5,0.5,0.5,1);
		    glPopMatrix();
		}
				
		glPopMatrix();
				
		glPopAttrib();
	    }
	}
    }
	
    glutSwapBuffers();
}

void idle(void)
{
    usleep(1./24*1e6);
	
    if (playdata.valid)
    {
	refinedpath.follow(playdata.frame, camera);
	playdata.frame = (playdata.frame + 1) % refinedpath.size();
	char buf[1024];
	sprintf(buf, "sshot%05d.tga", playdata.frame);
	//gltWriteTGA(buf);
    }
	
    glutPostRedisplay();
}

void keyboard(unsigned char k, int x, int y)
{
    std::cout << "keyboard\n" ; 
    const double stepsize = 3*camera.suggested_stepsize();
    assert(stepsize >0);
    switch (k) {
    case 'q':
	exit(0);
	break;
    case 'w':
	camera.step(stepsize, 2);
	break;
    case 's':
	camera.step(-stepsize, 2);
	break;
    case 'a':
	camera.step(stepsize, 0);
	break;
    case 'd':
	camera.step(-stepsize, 0);
	break;
    case 'c':
	camera.step(stepsize, 1);
	break;
    case ' ':
	camera.step(-stepsize, 1);
	break;
    case 'z':
	camera.zero_pitch();
	break;
    case 'b':
    {
	/*GLdouble VP[4] = {0, 0, 0, 0};
			
	  glGetDoublev(GL_VIEWPORT, VP);
			
	  float depth = -1;
	  glReadPixels(x, VP[3] - y, 1, 1, GL_DEPTH_COMPONENT, GL_FLOAT, &depth);
			
	  printf("depth is %f\n", depth);
			
	  testpos0 = camera.backproject(x, y, depth);
	  testpos1 = camera.backproject(x, y, 0);
	*/
	if (editdata.selected)
	{
	    path.stopping_flip();
	    refinedpath = path.refine_uniform(1e6);
	}
			
	break;
    }
    case 'u':
	path.undo();
	break;
    case 'p':
	static GLdouble MV[16];

	playdata.valid = (bool)(1 - (int)playdata.valid);
	playdata.frame = 0;
			
	if (playdata.valid)
	    glGetDoublev(GL_MODELVIEW_MATRIX, MV); //save it for later
	else 
	    glLoadMatrixd(MV);
			
	break;
    case 'r':
	printf("smoothing the refined path by %d steps\n", 10);
	refinedpath.smooth(10);
	break;
    case 'f':
	printf("finalizing...");
	path.save(camera, "tmp-camera.mv");
	refinedpath.save(camera);
	printf("done.\n");
	break;
    default:
	break;
    }
}

void motion(int x, int y)
{
    std::cout << "motion" << x << ", " << y << "\n" ; 
	
    if (trackballaxis.valid)
    {
	double xp = x - trackballaxis.xpixelcenter;
	double yp = -(y - trackballaxis.ypixelcenter);
	double zp = sqrt(std::max(0.,pow(trackballaxis.pixelradius,2) - xp * xp - yp * yp)); //std::max(0., trackballaxis.pixelradius - sqrt(xp * xp + yp * yp));
		
	double azi, ele;
	trackballaxis.update(xp, yp, zp, azi, ele);
	//trackballaxis.xpixelcenter = x;
	//trackballaxis.ypixelcenter = y;
		
	trackballaxis.xref = xp;
	trackballaxis.yref = yp;
	trackballaxis.zref = zp;
		
	double x,y,z, a,e;
	path.get_selected(x,y,z, a,e);
	path.set_selected(x,y,z, a + azi, e - ele);
		
	//refinedpath = path.refine_uniform();

    }
    else	if (editdata.selected)
    {
	SmartCamera::Pos p = camera.backproject(x, y, editdata.measured_depth);

	path.redisplace(p.x, p.y, p.z);
	refinedpath = path.refine_uniform(1e3);

    }
    else 
	if (navigationdata.refvalid)
	{
	    camera.mouse_control_constrained(x, y, navigationdata.xref, navigationdata.yref, navigationdata.zref);
	    navigationdata.xref = x;
	    navigationdata.yref = y;
	}
	

	
    glutPostRedisplay();
}

void mouse(int button, int state, int x, int y)
{
    std::cout << "mouse\n" ;
	
    //we need this in almost all cases
    GLdouble VP[4] = {0, 0, 0, 0};
    glGetDoublev(GL_VIEWPORT, VP);
	
    if(button == GLUT_RIGHT_BUTTON) //right button is camera control
    {	
	if(state == GLUT_DOWN)
	{
	    navigationdata.refvalid = true;
	    navigationdata.xref = x;
	    navigationdata.yref = y;
			
	    glReadPixels(navigationdata.xref, VP[3] - navigationdata.yref, 1, 1, GL_DEPTH_COMPONENT, GL_FLOAT, &navigationdata.zref);
	}
	else 
	    navigationdata.refvalid = false;
    }
    else if (button == GLUT_LEFT_BUTTON) 
    {
	// left button is control point redisplacement 
	// unless the mouse is located in the orientation-editing subwindow
	const double d = sqrt(pow(x - trackballaxis.xpixelcenter,2) + pow(y-trackballaxis.ypixelcenter,2)) - trackballaxis.pixelradius;
		
	if (d < 0 && trackballaxis.valid)
	{
	    if(state == GLUT_DOWN)
	    {
		double xref = x - trackballaxis.xpixelcenter;
		double yref = -(y - trackballaxis.ypixelcenter);
		double zref = sqrt(std::max(0.,pow(trackballaxis.pixelradius,2) - xref * xref - yref * yref)); 
		trackballaxis.valid = (state == GLUT_DOWN);
		trackballaxis.setref(xref, yref, zref);
				
		return;
	    }
	    else 
	    {
		refinedpath = path.refine_uniform(1e6);
		return;
	    }
	}
		
	if(state == GLUT_DOWN)
	{
	    //lets select the closest control point, if it is not too far
			
	    glReadPixels(x, VP[3] - y, 1, 1, GL_DEPTH_COMPONENT, GL_FLOAT, &editdata.measured_depth);
			
	    SmartCamera::Pos p = camera.backproject(x, y, editdata.measured_depth);
			
	    std::vector<double> xs, ys, zs;
	    path.get(xs, ys, zs);
	    const int isel = camera.closest2pixel(x, y, editdata.measured_depth, 20, xs, ys, zs);
			
	    path.select(isel);
	    editdata.selected = isel>=0;
	}
	else 
	{
	    refinedpath = path.refine_uniform(1e6);
	    //editdata.selected = false;
	}
    }
    else if (button == GLUT_MIDDLE_BUTTON) 
    {
	if(state == GLUT_DOWN)
	{
	    //select a control point, then open the control-editing subwindow
	    float depth;
	    glReadPixels(x, VP[3] - y, 1, 1, GL_DEPTH_COMPONENT, GL_FLOAT, &depth);
			
	    SmartCamera::Pos p = camera.backproject(x, y, depth);
			
	    std::vector<double> xs, ys, zs;
	    path.get(xs, ys, zs);
	    const int isel = camera.closest2pixel(x, y, depth, 20, xs, ys, zs);
	    path.select(isel);
			
	    static int oldselected = -1;
	    trackballaxis.valid = oldselected != isel && isel>=0;
	    oldselected = trackballaxis.valid ? isel : -1;
	}
	else
	    refinedpath = path.refine_uniform(1e6);
    }
}

int main (int argc,  char ** const argv) 
{
    if (argc < 3)
    {
	std::cout << "Usage campath <file> <number-of-frames> <path/to/cloud1> <path/to/cloud2> ...\n";
	exit(-1);
    }

    const char * filename = argv[1];
    printf("camera path: <%s>\n", filename);
    path = EditablePath(filename);
    referencepath = (Path)path;
    referencepath.set_tube_colors(0.6, 0.6, 0.8);
    desired_frames = atoi(argv[2]);

    for(int i = 3; i < argc; ++i)
    {
	printf("loading cloud path <%s>\n", argv[i]);
	std::ifstream culo(argv[i]);
	assert(culo.good());
	pointcloud.load_from_file(culo);
    }
	
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_DEPTH | GLUT_STENCIL | GLUT_MULTISAMPLE);
    glutInitWindowSize(1280, 720);
    glutCreateWindow("Smooth camera");

    glHint(GL_POLYGON_SMOOTH_HINT, GL_NICEST);
    glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST);
	
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    const double kx = 0.005;// * 1.42;
    const double ky = kx / 1280*720.;
    glFrustum(-kx, kx, -ky, ky, 0.02, 10);

    {
	FILE * f = fopen("curr-P.txt", "w");

	float P[16];
	glGetFloatv(GL_PROJECTION_MATRIX, P);

	for(int i = 0; i < 16; ++i)
	    fprintf(f, "%g, ", P[i]);
	fprintf(f, "\n");
	fclose(f);
    }

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
	
    //light moves with the viewer
    {
	GLfloat lightdir[4] = { -0.5, -0.5, 1, 0};
	glLightfv(GL_LIGHT0, GL_POSITION, lightdir);
    }

    //glLightfv(GL_LIGHT0, GL_DIFFUSE, white);
    //glLightfv(GL_LIGHT0, GL_SPECULAR, white);
    //glLightModelfv(GL_LIGHT_MODEL_AMBIENT, light_ambient);
	
    refinedpath = path.refine_uniform(1e6);
    camera.move_to(0.75,1,.99);
    double bbox[2][3];
    refinedpath.boundingbox(bbox);
    camera.freely_approach(bbox);
    camera.zero_pitch();

    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LESS);
    glDepthMask(GL_TRUE);
	
    glutDisplayFunc(display);
    glutIdleFunc(idle);
    glutKeyboardFunc(keyboard);
    glutMouseFunc(mouse);
    glutMotionFunc(motion);
    glutMainLoop();
	
    return 0;
}
