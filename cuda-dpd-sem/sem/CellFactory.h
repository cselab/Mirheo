#pragma once

struct CellParams
{
	float rcutoff, gamma, u0, rho, req, D, rc;
};

void produceCell(int n, float * xyz, CellParams& params);

