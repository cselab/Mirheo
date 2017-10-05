#pragma once

/**
 * GPU-compatible struct of all the relevant data
 */
struct REOVview : public ROVview
{
	float3 axes    = {0,0,0};
	float3 invAxes = {0,0,0};
};

static REOVview create_REOVview(RigidEllipsoidObjectVector* reov, LocalRigidEllipsoidObjectVector* lreov)
{
	// Create a default view
	REOVview view;
	if (reov == nullptr || lreov == nullptr)
		return view;

	view.ROVview::operator= ( create_ROVview(reov, lreov) );

	// More fields
	view.axes = reov->axes;
	view.invAxes = 1.0 / view.axes;

	return view;
}
