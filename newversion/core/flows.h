// This macro is here because of c++11 and c++14 limitations with cuda
// hopefully later on they will be alleviated
// Needed generic lambdas and no scope restriction on __device__ lambdas
//
// Parameter of this macro is a cuda kernel that will be called "as is"
// It has to have a templated parameter integrate, which is a lambda function:
// integrate(float4& x, float4& v, const float4 a, const int pid)
//
// It has to integrate coordinates and velocities and apply all the other transformations
//
// Available vars: pv:     ParticleVector
// 			       config: IniParser with settings
//				   dt:     timestep
//
// This command will generate macro (add slashes to the end of lines):
// perl -pi -e 's/(.*)/$1 \\/ unless (/^\s*\/\// or /\s*\\$/);' flows.h
// You may need to manually remove a few trailing slashes
//
#define flowMacroWrapper(function) \
do { \
	const std::string flowType = config.getString("FlowField", "type", "noflow"); \
 \
	if (flowType == "noflow") \
	{ \
		auto integrate = [dt] __device__ (float4& x, float4& v, const float4 a, const int pid) { \
			v.x += a.x*dt; \
			v.y += a.y*dt; \
			v.z += a.z*dt; \
 \
			x.x += v.x*dt; \
			x.y += v.y*dt; \
			x.z += v.z*dt; \
		}; \
 \
		function; \
	} \
 \
	if (flowType == "const_grad_P") \
	{ \
		float3 additionalForce = config.getFloat3("FlowField", "AdditionalForce"); \
		auto integrate = [dt, additionalForce] __device__ (float4& x, float4& v, const float4 a, const int pid) { \
			v.x += (a.x+additionalForce.x) * dt; \
			v.y += (a.y+additionalForce.y) * dt; \
			v.z += (a.z+additionalForce.z) * dt; \
 \
			x.x += v.x*dt; \
			x.y += v.y*dt; \
			x.z += v.z*dt; \
		}; \
 \
		function; \
	} \
} while(0)
