/*
 *  SmartCamera.h
 *  SmoothCameraPath
 *
 *  Created by Diego Rossinelli on 9/18/12.
 *  Copyright 2012 ETH Zurich. All rights reserved.
 *
 */
#pragma once


#include <cmath>
#include <cassert>
#include <cstring>
#include <vector>
#include <algorithm>

#include <GL/gl.h>

class SmartCamera
{
public:

	struct Pos { double x, y, z; };

private:

	struct Basis { 
		double cx0, cx1, cx2;
		double cy0, cy1, cy2;
		double cz0, cz1, cz2;
	};
	
	Pos _pos(const GLdouble MV[4][4])
	{
		Pos retval;
		
		retval.x = -(MV[0][0] * MV[3][0] + MV[0][1] * MV[3][1] + MV[0][2] * MV[3][2]);
		retval.y = -(MV[1][0] * MV[3][0] + MV[1][1] * MV[3][1] + MV[1][2] * MV[3][2]);
		retval.z = -(MV[2][0] * MV[3][0] + MV[2][1] * MV[3][1] + MV[2][2] * MV[3][2]);
		
		return retval;
	}
	
	Pos _transform(const double x, const  double y, const double z, const GLdouble MV[4][4])
	{
		const double p[4] = {x, y, z, 1};
		
		double r[3] = {0, 0, 0};
		
		for(int d = 0; d < 3; d++)
			for(int c = 0; c < 4; c++)
				r[d] += MV[c][d] * p[c];
		
		const Pos retval = {r[0], r[1], r[2]};
		
		return retval;
	};
	
	static Basis _basis(const double azimuth, const double elevation)
	{
		/* 
		 //ORIGINAL CODE 
		 Basis ref = {
			cos(azimuth + M_PI / 2), 0, sin(azimuth + M_PI / 2),
			cos(elevation + M_PI / 2) * cos(azimuth), sin(elevation + M_PI / 2), cos(elevation + M_PI / 2) * sin(azimuth),
			cos(elevation + M_PI) * cos(azimuth), sin(elevation + M_PI) , cos(elevation + M_PI) * sin(azimuth)
		};
		*/
		
		Basis retval = {
			-sin(azimuth), 0, cos(azimuth),
			-sin(elevation) * cos(azimuth), cos(elevation), -sin(elevation) * sin(azimuth),
			-cos(elevation) * cos(azimuth), -sin(elevation) , -cos(elevation) * sin(azimuth)
		};
		
		return retval;
	}

	void _projection_info(double& left, double& right, double& bottom, double& top, double& near, double& far, const GLdouble P[4][4])
	{
		assert(P[3][3] == 0.0);
		
		near =			P[3][2] / (-1 + P[2][2]); 
		far =			P[3][2] / (+1 + P[2][2]);
		left =			near * (-1 + P[2][0]) / P[0][0];
		right =			near * (+1 + P[2][0]) / P[0][0];
		bottom =		near * (-1 + P[2][1]) / P[1][1];
		top =			near * (+1 + P[2][1]) / P[1][1];		
	}
	
	void _point_to(const double azimuth, const double elevation, GLdouble MV[4][4])
	{	
		const Pos pos = _pos(MV);

		const Basis b = _basis(azimuth, elevation);
				
		MV[0][0] = b.cx0;		MV[1][0] = b.cx1;		MV[2][0] = b.cx2;
		MV[0][1] = b.cy0;		MV[1][1] = b.cy1;		MV[2][1] = b.cy2;
		MV[0][2] = b.cz0;		MV[1][2] = b.cz1;		MV[2][2] = b.cz2;
		
		MV[3][0] = -(pos.x * b.cx0 + pos.y * b.cx1 + pos.z * b.cx2);
		MV[3][1] = -(pos.x * b.cy0 + pos.y * b.cy1 + pos.z * b.cy2);
		MV[3][2] = -(pos.x * b.cz0 + pos.y * b.cz1 + pos.z * b.cz2);
		
		MV[0][3] = MV[1][3] = MV[2][3] = 0;		MV[3][3] = 1;		
	}
	
	std::vector<double> _nice(std::vector<double> angles)
	{
		std::vector<double> retval;
		
		const double x0 = angles.front();
		
		for(int i=1; i<angles.size(); ++i)
		{
			const double x = angles[i];
			const double k = floor( 0.5 + (x0 - x) / (2 * M_PI) );
			
			retval.push_back(x + k * 2 * M_PI);
		}
		
		return retval;
	}
	
	void _point_to(const std::vector<double> xs, const std::vector<double> ys, const std::vector<double> zs, GLdouble MV[4][4])
	{
		Pos pos = _pos(MV);

		std::vector<double> azimuths, elevations;
		
		for(int i=0; i<xs.size(); ++i)
		{
			Pos xt = {
				xs[i] - pos.x, 
				ys[i] - pos.y, 
				zs[i] - pos.z
			}; 

			azimuths.push_back( atan2(xt.z, xt.x));
			elevations.push_back( atan2(xt.y, sqrt(xt.x * xt.x + xt.z * xt.z)) );					
		}
		
		azimuths = _nice(azimuths);
		
		const double azimin = *std::min_element(azimuths.begin(), azimuths.end());
		const double azimax = *std::max_element(azimuths.begin(), azimuths.end());

		const double elemin = *std::min_element(elevations.begin(), elevations.end());
		const double elemax = *std::max_element(elevations.begin(), elevations.end());

		_point_to((azimin + azimax) / 2, (elemin + elemax)/2, MV);
	}
	
	void _free_point_to(const std::vector<double> xs, const std::vector<double> ys, const std::vector<double> zs, GLdouble MV[4][4])
	{
		std::vector<double> azimuths, elevations;
		
		for(int i=0; i<xs.size(); ++i)
		{
			Pos xt = _transform(xs[i], ys[i], zs[i], MV);
			
			azimuths.push_back( atan2(xt.z, xt.x));
			elevations.push_back( atan2(xt.y, sqrt(xt.x * xt.x + xt.z * xt.z)) );					
		}
		
		azimuths = _nice(azimuths);
		
		const double azimin = *std::min_element(azimuths.begin(), azimuths.end());
		const double azimax = *std::max_element(azimuths.begin(), azimuths.end());
		
		const double elemin = *std::min_element(elevations.begin(), elevations.end());
		const double elemax = *std::max_element(elevations.begin(), elevations.end());
		
		Basis b = _basis((azimin + azimax) / 2, (elemin + elemax) / 2);
		
		const double C[3][3] = {
			b.cx0, b.cy0, b.cz0, 
			b.cx1, b.cy1, b.cz1, 
			b.cx2, b.cy2, b.cz2 
		};
		
		const double OLDC[3][3] = {
			MV[0][0], MV[0][1], MV[0][2],
			MV[1][0], MV[1][1], MV[1][2],
			MV[2][0], MV[2][1], MV[2][2]
		};
		
		Pos pos = _pos(MV);
		
		for(int dr = 0 ; dr < 3 ; ++dr)
			for(int dc = 0 ; dc < 3 ; ++dc)
			{
				MV[dc][dr] = 0;
				
				for(int k = 0; k < 3 ; ++k)
					MV[dc][dr] += C[k][dr] * OLDC[dc][k];
			}
		
		MV[3][0] = -(pos.x * MV[0][0] + pos.y * MV[1][0] + pos.z * MV[2][0]);
		MV[3][1] = -(pos.x * MV[0][1] + pos.y * MV[1][1] + pos.z * MV[2][1]);
		MV[3][2] = -(pos.x * MV[0][2] + pos.y * MV[1][2] + pos.z * MV[2][2]);
		
		MV[0][3] = MV[1][3] = MV[2][3] = 0;		MV[3][3] = 1;
	}
	
	void _zadjust(const std::vector<double> xs, const std::vector<double> ys, const std::vector<double> zs, GLdouble MV[4][4], GLdouble P[4][4])
	{
		std::vector<double> leaps;
				
		double l, r, b, t, n, f;
		_projection_info(l, r, b, t, n, f, P);
		
		for(int i=0; i<xs.size(); ++i)
		{
			Pos eye = _transform(xs[i], ys[i], zs[i], MV);
			
			leaps.push_back(-fabs(eye.x / l * n) - eye.z);
			leaps.push_back(-fabs(eye.x / r * n) - eye.z);
			leaps.push_back(-fabs(eye.y / b * n) - eye.z);
			leaps.push_back(-fabs(eye.y / t * n) - eye.z);
		}
		
		MV[3][2] += *std::min_element(leaps.begin(), leaps.end());
	}
	
	Pos _backproject_eye(const double x, const double y, const float depth, const GLdouble VP[4], GLdouble P[4][4])
	{
		GLdouble invP[4][4];
		
		double l, r, b, t, n, f;		
		_projection_info(l, r, b, t, n, f, P);
		
		memset((GLdouble *)invP, 0, sizeof(GLdouble)*4*4);
		
		invP[0][0] = (r - l) / (2 * n);
		invP[3][0] = (r + l) / (2 * n);
		invP[1][1] = (t - b) / (2 * n);
		invP[3][1] = (t + b) / (2 * n);
		invP[3][2] = -1;
		invP[2][3] = -(f - n) / (2 * f * n);
		invP[3][3] = +(f + n) / (2 * f * n);
		
		const double x_di = -1 + 2 * (x + 0.5 - VP[0]) / VP[2];
		const double y_di = -(-1 + 2 * (y + 0.5 - VP[1]) / VP[3]);
		const double z_di = -1 + 2 * depth;
		
		const double w_c = 2 * f * n / (f + n - z_di * (f - n));
		const double x_c = x_di * w_c;
		const double y_c = y_di * w_c;
		const double z_c = z_di * w_c;
		
		const double x_e = invP[0][0] * x_c + invP[1][0] * y_c + invP[2][0] * z_c + invP[3][0] * w_c;
		const double y_e = invP[0][1] * x_c + invP[1][1] * y_c + invP[2][1] * z_c + invP[3][1] * w_c;
		const double z_e = invP[0][2] * x_c + invP[1][2] * y_c + invP[2][2] * z_c + invP[3][2] * w_c;
		const double w_e = invP[0][3] * x_c + invP[1][3] * y_c + invP[2][3] * z_c + invP[3][3] * w_c;
		
		assert(fabs(w_e - 1) < 1e-8);
		
		Pos retval = {x_e, y_e, z_e};
		
		return retval;
	}
	
	Pos _backproject(const double x, const double y, const float depth, const GLdouble VP[4], const GLdouble MV[4][4], GLdouble P[4][4])
	{
		Pos eye_coord = _backproject_eye(x, y, depth, VP, P);
		
		Pos pos = _pos(MV);

		const double x_w = MV[0][0] * eye_coord.x + MV[0][1] * eye_coord.y + MV[0][2] * eye_coord.z +  pos.x ;
		const double y_w = MV[1][0] * eye_coord.x + MV[1][1] * eye_coord.y + MV[1][2] * eye_coord.z +  pos.y ;
		const double z_w = MV[2][0] * eye_coord.x + MV[2][1] * eye_coord.y + MV[2][2] * eye_coord.z +  pos.z ;
		
		Pos retval = {x_w, y_w, z_w};
		
		return retval;
	}
	
	Pos _project(const double x, const double y, const double z, const GLdouble MV[4][4], GLdouble P[4][4])
	{
		//returns device-independent coords
		
		const double x_e = MV[0][0] * x + MV[1][0] * y + MV[2][0] * z +  MV[3][0] ;
		const double y_e = MV[0][1] * x + MV[1][1] * y + MV[2][1] * z +  MV[3][1] ;
		const double z_e = MV[0][2] * x + MV[1][2] * y + MV[2][2] * z +  MV[3][2] ;
		
		const double x_c = P[0][0] * x_e + P[1][0] * y_e + P[2][0] * z_e +  P[3][0];
		const double y_c = P[0][1] * x_e + P[1][1] * y_e + P[2][1] * z_e +  P[3][1];
		const double z_c = P[0][2] * x_e + P[1][2] * y_e + P[2][2] * z_e +  P[3][2];
		const double w_c = P[0][3] * x_e + P[1][3] * y_e + P[2][3] * z_e +  P[3][3];
		
		Pos retval = { x_c / w_c, y_c / w_c, z_c / w_c };
		
		return retval;
	}
	
public:
	
	void step(const double step, const int iaxis)
	{
		GLdouble MV[4][4];
		
		glGetDoublev(GL_MODELVIEW_MATRIX, (GLdouble *)MV);
		
		MV[3][iaxis] += step;
		
		glLoadMatrixd((GLdouble *)MV);

		double * data = (double *)MV;
		FILE * f = fopen("cam-mv.txt", "w");
		
		fprintf(f, "-MV=");
		for(int i = 0; i < 16; ++i)
		    fprintf(f, "%f%s", data[i], (i == 15 ? "\n" : ", "));
		
		fclose(f);
	}
	
	void move_to(const double x, const double y, const double z)
	{
		GLdouble MV[4][4];
		
		glGetDoublev(GL_MODELVIEW_MATRIX, (GLdouble *)MV);
		
		MV[3][0] = -(MV[0][0] * x + MV[1][0] * y + MV[2][0] * z);
		MV[3][1] = -(MV[0][1] * x + MV[1][1] * y + MV[2][1] * z);
		MV[3][2] = -(MV[0][2] * x + MV[1][2] * y + MV[2][2] * z);
		
		glLoadMatrixd((GLdouble *)MV);
	}
	
	void point_to(const double azimuth, const double elevation)
	{
		GLdouble MV[4][4];
		
		glGetDoublev(GL_MODELVIEW_MATRIX, (GLdouble *)MV);
		
		_point_to(azimuth, elevation, MV);
		
		glLoadMatrixd((GLdouble *)MV);
	}

	void point_to(const double x, const double y, const double z)
	{
		GLdouble MV[4][4];
		
		glGetDoublev(GL_MODELVIEW_MATRIX, (GLdouble *)MV);
		
		Pos pos = _pos(MV);
		
		const double dx = x - pos.x;
		const double dy = y - pos.y;
		const double dz = z - pos.z;
		
		const double azimuth = atan2(dz, dx);
		const double elevation = atan2(dy, sqrt(dx * dx + dz * dz));
		
		_point_to(azimuth, elevation, MV);
		
		glLoadMatrixd((GLdouble *)MV);
	}
	
	void point_to(double bounding_box[2][3])
	{
		std::vector<double> xs, ys, zs;
		
		for(int iz=0; iz<2; ++iz)
			for(int iy=0; iy<2; ++iy)
				for(int ix=0; ix<2; ++ix)
				{
					xs.push_back(bounding_box[ix][0]);
					ys.push_back(bounding_box[iy][1]);
					zs.push_back(bounding_box[iz][2]);
				}
		
		GLdouble MV[4][4];
		
		glGetDoublev(GL_MODELVIEW_MATRIX, (GLdouble *)MV);

		_point_to(xs, ys, zs, MV);
		
		glLoadMatrixd((GLdouble *)MV);
	}
	
	void freely_approach(double bounding_box[2][3])
	{
		std::vector<double> xs, ys, zs;
		
		for(int iz=0; iz<2; ++iz)
			for(int iy=0; iy<2; ++iy)
				for(int ix=0; ix<2; ++ix)
				{
					xs.push_back(bounding_box[ix][0]);
					ys.push_back(bounding_box[iy][1]);
					zs.push_back(bounding_box[iz][2]);
				}
		
		GLdouble MV[4][4], P[4][4];
		
		glGetDoublev(GL_MODELVIEW_MATRIX, (GLdouble *)MV);
		glGetDoublev(GL_PROJECTION_MATRIX, (double *)&P);
		
		for(int i=0; i<1; ++i)
		{
			_free_point_to(xs, ys, zs, MV);
			_zadjust(xs, ys, zs, MV, P);
		}
		
		_free_point_to(xs, ys, zs, MV);
		
		glLoadMatrixd((GLdouble *)MV);
	}
	
	void zadjust(double x, double y, double z, double r)
	{
		std::vector<double> xs, ys, zs;
		
		GLdouble MV[4][4], P[4][4];
		
		glGetDoublev(GL_MODELVIEW_MATRIX, (GLdouble *)MV);
		glGetDoublev(GL_PROJECTION_MATRIX, (double *)&P);
		
		xs.push_back(x + r * MV[0][2]);
		ys.push_back(y + r * MV[1][2]);
		zs.push_back(z + r * MV[2][2]);
		
		_zadjust(xs, ys, zs, MV, P);
		
		glLoadMatrixd((GLdouble *)MV);
	}
	
	double suggested_stepsize()
	{
		GLdouble P[4][4];
		
		glGetDoublev(GL_PROJECTION_MATRIX, (double *)&P);
		
		double l, r, b, t, n, f;		
		_projection_info(l, r, b, t, n, f, P);
		
		return n*0.333;
	}
		
	void mouse_control(const double x, const double y, const int xref, const int yref, const float depth)
	{
		GLdouble VP[4] = {0, 0, 0, 0};
		GLdouble MV[4][4], P[4][4];
		
		glGetDoublev(GL_VIEWPORT, VP);
		glGetDoublev(GL_MODELVIEW_MATRIX, (GLdouble *)MV);
		glGetDoublev(GL_PROJECTION_MATRIX, (double *)&P);
				
		Pos p = _backproject_eye(x, y, 0, VP, P);
		const double desiredazimuth = atan2(p.x, -p.z);
		const double desiredelevation = atan2(p.y, sqrt(p.x * p.x + p.z * p.z));

		Pos pref = _backproject_eye(xref, yref, depth, VP, P);
		const double startazimuth = atan2(pref.x, -pref.z);
		const double startelevation = atan2(pref.y, sqrt(pref.x * pref.x + pref.z * pref.z));
		
		const double azimuth = -startazimuth + desiredazimuth;
		const double elevation = -startelevation + desiredelevation;
		const Basis basis = _basis(-azimuth -M_PI/2, -elevation);
		
		const double C[3][3] = {
			basis.cx0, basis.cy0, basis.cz0, 
			basis.cx1, basis.cy1, basis.cz1, 
			basis.cx2, basis.cy2, basis.cz2 
		};
		
		const double OLDC[3][3] = {
			MV[0][0], MV[0][1], MV[0][2],
			MV[1][0], MV[1][1], MV[1][2],
			MV[2][0], MV[2][1], MV[2][2]
		};
		
		Pos pos = _pos(MV);
		
		for(int dr = 0 ; dr < 3 ; ++dr)
			for(int dc = 0 ; dc < 3 ; ++dc)
			{
				MV[dc][dr] = 0;
				
				for(int k = 0; k < 3 ; ++k)
					MV[dc][dr] += C[k][dr] * OLDC[dc][k];
			}
		
		MV[3][0] = -(pos.x * MV[0][0] + pos.y * MV[1][0] + pos.z * MV[2][0]);
		MV[3][1] = -(pos.x * MV[0][1] + pos.y * MV[1][1] + pos.z * MV[2][1]);
		MV[3][2] = -(pos.x * MV[0][2] + pos.y * MV[1][2] + pos.z * MV[2][2]);
		
		MV[0][3] = MV[1][3] = MV[2][3] = 0;		MV[3][3] = 1;
		
		glLoadMatrixd((GLdouble *)MV);
	}
	
	void zero_pitch()
	{
		GLdouble MV[4][4];
		
		glGetDoublev(GL_MODELVIEW_MATRIX, (GLdouble *)MV);
		
		const double cz0 = -MV[0][2];
		const double cz1 = -MV[1][2];
		const double cz2 = -MV[2][2];
		
		const double azimuth = atan2(cz2, cz0);
		const double elevation = atan2(cz1, sqrt(cz0 * cz0 + cz2 * cz2));
		
		point_to(azimuth, elevation);
	}
	
	void mouse_control_constrained(const double x, const double y, const int xref, const int yref, const float depth)
	{		
		GLdouble VP[4] = {0, 0, 0, 0};
		GLdouble MV[4][4], P[4][4];
		
		glGetDoublev(GL_VIEWPORT, VP);
		glGetDoublev(GL_MODELVIEW_MATRIX, (GLdouble *)MV);
		glGetDoublev(GL_PROJECTION_MATRIX, (double *)&P);

		Pos t0 = _backproject(x, y, 0, VP, MV, P);
		Pos t1 = _backproject(x, y, 1, VP, MV, P);
		
		Pos pcam = _pos(MV);
		Pos pref = _backproject(xref, yref, depth, VP, MV, P);
				
		const double rescale1 = 1./sqrt(pow(t1.x - t0.x, 2) + pow(t1.y - t0.y, 2) + pow(t1.z - t0.z, 2));
		const double xdir1 = rescale1 * (t1.x - t0.x);
		const double ydir1 = rescale1 * (t1.y - t0.y);
		const double zdir1 = rescale1 * (t1.z - t0.z);
		
		const double rescale2 = 1./sqrt(pow(pref.x - pcam.x, 2) + pow(pref.y - pcam.y, 2) + pow(pref.z - pcam.z, 2));
		const double xdir2 = rescale2 * (pref.x - pcam.x);
		const double ydir2 = rescale2 * (pref.y - pcam.y);
		const double zdir2 = rescale2 * (pref.z - pcam.z);
		
					
		const double azimuthdelta = (atan2(zdir1, xdir1) - atan2(zdir2, xdir2));
		const double elevationdelta = atan2(ydir1, sqrt(xdir1*xdir1 + zdir1*zdir1)) - atan2(ydir2, sqrt(xdir2*xdir2 + zdir2*zdir2));
		
		const double azimuth = atan2(-MV[2][2], -MV[0][2]);
		const double elevation = atan2(-MV[1][2], sqrt(MV[0][2]*MV[0][2] + MV[2][2]*MV[2][2]));
			
		point_to(azimuth - azimuthdelta, elevation - elevationdelta);
	}
	
	Pos backproject(const double x, const double y, const float depth)
	{
		GLdouble VP[4] = {0, 0, 0, 0};
		GLdouble MV[4][4], P[4][4];
		
		glGetDoublev(GL_VIEWPORT, VP);
		glGetDoublev(GL_MODELVIEW_MATRIX, (GLdouble *)MV);
		glGetDoublev(GL_PROJECTION_MATRIX, (double *)&P);
		
		return _backproject(x, y, depth, VP, MV, P);
	}
	
	Pos project2di(const double x, const double y, const double z)
	{
		GLdouble MV[4][4], P[4][4];
		
		glGetDoublev(GL_MODELVIEW_MATRIX, (GLdouble *)MV);
		glGetDoublev(GL_PROJECTION_MATRIX, (double *)P);
		
		return _project(x, y, z, MV, P);
	}	
	
	Pos project2pixels(const double x, const double y, const double z)
	{
		GLdouble MV[4][4], P[4][4];
		glGetDoublev(GL_MODELVIEW_MATRIX, (GLdouble *)MV);
		glGetDoublev(GL_PROJECTION_MATRIX, (double *)P);
		
		Pos p = _project(x, y, z, MV, P);
		
		GLdouble VP[4] = {0, 0, 0, 0};
		glGetDoublev(GL_VIEWPORT, VP);

		Pos retval = { 
			VP[0] + VP[2] * (0.5 + p.x * 0.5),
			VP[1] + VP[3] * (0.5 - p.y * 0.5),
			0.5 + p.z * 0.5
		};
	
		return retval;
	}
	
	void points2pixels(const std::vector<double> xs, const std::vector<double> ys, const std::vector<double> zs,
					   std::vector<double>& xpixels, std::vector<double>& ypixels, std::vector<double>& depthpixels)
	{
		xpixels.clear();
		ypixels.clear();
		depthpixels.clear();
		
		const int N = xs.size();
		for(int i=0; i<N; ++i)
		{
			Pos p = project2pixels(xs[i], ys[i], zs[i]);
						
			xpixels.push_back(p.x);
			ypixels.push_back(p.y);
			depthpixels.push_back(p.z);
		}
	}
	
	int closest2pixel(const int x, const int y, const double depth, const double distance_threshold, std::vector<double> xs, std::vector<double> ys, std::vector<double> zs)
	{
		const int N = xs.size();
		
		std::vector<double> ds;
		for(int i=0; i<N; ++i)
		{
			Pos p = project2pixels(xs[i], ys[i], zs[i]);
			
			const double d = sqrt(pow(p.x - x,2) + pow(p.y - y,2) + 0*pow(p.z - depth, 2));
			
			ds.push_back(d);
		}
		
		if (*min_element(ds.begin(), ds.end()) > distance_threshold) return -1;
		
		return min_element(ds.begin(), ds.end()) - ds.begin();
	}
	
	void frame(double cx[3], double cy[3], double cz[3])
	{
		GLdouble MV[4][4];
		glGetDoublev(GL_MODELVIEW_MATRIX, (GLdouble *)MV);
		
		for(int i=0; i<3; ++i) cx[i] = MV[i][0];
		for(int i=0; i<3; ++i) cy[i] = MV[i][1];
		for(int i=0; i<3; ++i) cz[i] = MV[i][2];
	}
};
