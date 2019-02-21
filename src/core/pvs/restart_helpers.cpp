#include "restart_helpers.h"

namespace RestartHelpers
{

void copyShiftCoordinates(const DomainInfo &domain, const std::vector<Particle> &parts, LocalParticleVector *local)
{
    local->resize(parts.size(), 0);

    for (int i = 0; i < parts.size(); i++) {
        auto p = parts[i];
        p.r = domain.global2local(p.r);
        local->coosvels[i] = p;
    }
}

void make_symlink(MPI_Comm comm, std::string path, std::string name, std::string fname)
{
    int rank;
    MPI_Check( MPI_Comm_rank(comm, &rank) );

    if (rank == 0) {
        std::string lnname = path + "/" + name + ".xmf";
        
        std::string command = "ln -f " + fname + ".xmf " + lnname;
        if ( system(command.c_str()) != 0 )
            error("Could not create link for checkpoint file of PV '%s'", name.c_str());
    }    
}

} // namespace RestartHelpers
