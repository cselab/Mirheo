#include "restart_helpers.h"

namespace restart_helpers
{
    void exchangeParticles(const DomainInfo &domain, MPI_Comm comm, std::vector<Particle> &parts)
    {
        int size;
        int dims[3], periods[3], coords[3];
        MPI_Check( MPI_Comm_size(comm, &size) );
        MPI_Check( MPI_Cart_get(comm, 3, dims, periods, coords) );

        MPI_Datatype ptype;
        MPI_Check( MPI_Type_contiguous(sizeof(Particle), MPI_CHAR, &ptype) );
        MPI_Check( MPI_Type_commit(&ptype) );

        // Find where to send the read particles
        std::vector<std::vector<Particle>> sendBufs(size);

        for (auto& p : parts) {
            int3 procId3 = make_int3(floorf(p.r / domain.localSize));

            if (procId3.x >= dims[0] || procId3.y >= dims[1] || procId3.z >= dims[2])
                continue;

            int procId;
            MPI_Check( MPI_Cart_rank(comm, (int*)&procId3, &procId) );
            sendBufs[procId].push_back(p);
        }

        // Do the send
        std::vector<MPI_Request> reqs(size);
        for (int i = 0; i < size; i++) {
            debug3("Sending %d paricles to rank %d", sendBufs[i].size(), i);
            MPI_Check( MPI_Isend(sendBufs[i].data(), sendBufs[i].size(), ptype, i, 0, comm, reqs.data()+i) );
        }

        // recv data
        parts.resize(0);
        for (int i = 0; i < size; i++) {
            MPI_Status status;
            int msize;
            std::vector<Particle> recvBuf;
        
            MPI_Check( MPI_Probe(MPI_ANY_SOURCE, 0, comm, &status) );
            MPI_Check( MPI_Get_count(&status, ptype, &msize) );

            recvBuf.resize(msize);

            debug3("Receiving %d particles from ???", msize);
            MPI_Check( MPI_Recv(recvBuf.data(), msize, ptype, status.MPI_SOURCE, 0, comm, MPI_STATUS_IGNORE) );

            parts.insert(parts.end(), recvBuf.begin(), recvBuf.end());
        }

        MPI_Check( MPI_Waitall(reqs.size(), reqs.data(), MPI_STATUSES_IGNORE) );
        MPI_Check( MPI_Type_free(&ptype) );
    }
    
    void copyShiftCoordinates(const DomainInfo &domain, const std::vector<Particle> &parts, LocalParticleVector *local)
    {
        local->resize(parts.size(), 0);

        for (int i = 0; i < parts.size(); i++) {
            auto p = parts[i];
            p.r = domain.global2local(p.r);
            local->coosvels[i] = p;
        }
    }
}
