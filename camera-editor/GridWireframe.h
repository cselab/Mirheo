#pragma once

struct GridWireframe
{
	const double xorigin, yorigin, zorigin;
	double xextent, yextent, zextent;
	const int LINESMIN;
	
	GridWireframe(const double xorigin, const double yorigin, const double zorigin, 
			 const double xextent, const double yextent, const double zextent, 
			 const int LINESMIN):
	xorigin(xorigin), yorigin(yorigin), zorigin(zorigin), 
	xextent(xextent), yextent(yextent), zextent(zextent), LINESMIN(LINESMIN)
	{
		//adjust extents to be multiple of spacing
		const double minextent = std::min( xextent, std::min( yextent, zextent));
		const double spacing = minextent / LINESMIN ;
		
		this->xextent = spacing * ceil(xextent / spacing);
		this->yextent = spacing * ceil(yextent / spacing);
		this->zextent = spacing * ceil(zextent / spacing);
	}
	
	void paint(const double luminance = 1.0, const double alpha  = 1.0)
	{
		const double minextent = std::min( xextent, std::min( yextent, zextent));
		const double spacing = minextent / LINESMIN ;
		
		glPushAttrib(GL_CURRENT_BIT);
		glColor4f(luminance, luminance, luminance, alpha);
		
		//z-aligned slices
		{
			const int N = (int) floor(zextent / spacing);
			for(int i=0; i <= N; i++)
			{
				glBegin(GL_LINE_LOOP);
				glVertex3f(xorigin, yorigin, zorigin + i * spacing);	
				glVertex3f(xorigin + xextent, yorigin, zorigin + i * spacing);	
				glVertex3f(xorigin + xextent, yorigin + yextent, zorigin + i * spacing);	
				glVertex3f(xorigin, yorigin + yextent, zorigin + i * spacing);
				glEnd();
			}
		}
		
		//y-aligned slices
		{
			const int N = (int) floor(yextent / spacing);
			for(int i=0; i <= N; i++)
			{
				glBegin(GL_LINE_LOOP);
				glVertex3f(xorigin, yorigin + i * spacing, zorigin );	
				glVertex3f(xorigin + xextent, yorigin + i * spacing, zorigin);	
				glVertex3f(xorigin + xextent, yorigin + i * spacing, zorigin + zextent);	
				glVertex3f(xorigin, yorigin + i * spacing, zorigin + zextent);
				glEnd();
			}
		}
		
		//x-aligned slices
		{
			const int N = (int) floor(xextent / spacing);
			for(int i=0; i <= N; i++)
			{
				glBegin(GL_LINE_LOOP);
				glVertex3f(xorigin + i * spacing, yorigin, zorigin);
				glVertex3f(xorigin + i * spacing, yorigin + yextent, zorigin);
				glVertex3f(xorigin + i * spacing, yorigin + yextent, zorigin + zextent);
				glVertex3f(xorigin + i * spacing, yorigin, zorigin + zextent);
				glEnd();
			}
		}
		
		glPopAttrib();
	}	
};
