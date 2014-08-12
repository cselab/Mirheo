#pragma once

struct ParamsSEM
{
	float rcutoff, gamma, u0, rho, req, D, rc;
};

void cell_factory(int n, float * xyz, ParamsSEM& params);

