#include <plugins/interface.h>
#include <core/containers.h>
#include <core/datatypes.h>
#include <plugins/write_xdmf.h>

#include <vector>
#include <array>

class UniformCartesianDumper : public PostprocessPlugin
{
private:
	XDMFDumper* dumper;
	std::string path;

	int3 nranks3D;
	int3 resolution;
	float3 h;

	std::vector<XDMFDumper::ChannelType> channelTypes;
	std::vector<std::string> channelNames;
	std::vector<std::vector<float>> channels;

public:
	UniformCartesianDumper(std::string name, std::string path);

	void deserialize(MPI_Status& stat) override;
	void handshake() override;

	~UniformCartesianDumper() {};
};
