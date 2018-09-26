#include "restart_helpers.h"

namespace restart_helpers
{
    static void sendData(const std::vector<std::vector<Particle>> &sendBufs, std::vector<MPI_Request> &reqs,
                         MPI_Comm comm, MPI_Datatype type)
    {
        for (int i = 0; i < sendBufs.size(); i++) {
            debug3("Sending %d paricles to rank %d", sendBufs[i].size(), i);
            MPI_Check( MPI_Isend(sendBufs[i].data(), sendBufs[i].size(), type, i, 0, comm, reqs.data()+i) );
        }
    }

    static void recvData(int size, std::vector<Particle> &all, MPI_Comm comm, MPI_Datatype type)
    {
        all.resize(0);
        for (int i = 0; i < size; i++) {
            MPI_Status status;
            int msize;
            std::vector<Particle> recvBuf;
        
            MPI_Check( MPI_Probe(MPI_ANY_SOURCE, 0, comm, &status) );
            MPI_Check( MPI_Get_count(&status, type, &msize) );

            recvBuf.resize(msize);

            debug3("Receiving %d particles from ???", msize);
            MPI_Check( MPI_Recv(recvBuf.data(), msize, type, status.MPI_SOURCE, 0, comm, MPI_STATUS_IGNORE) );

            all.insert(all.end(), recvBuf.begin(), recvBuf.end());
        }
    }

    template <typename Splitter>
    static void exchangeParticles(const DomainInfo &domain, MPI_Comm comm, std::vector<Particle> &parts, Splitter splitter)
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

        splitter(dims, parts, sendBufs);

        std::vector<MPI_Request> reqs(size);
        
        sendData(sendBufs, reqs, comm, ptype);
        recvData(size, parts, comm, ptype);

        MPI_Check( MPI_Waitall(reqs.size(), reqs.data(), MPI_STATUSES_IGNORE) );
        MPI_Check( MPI_Type_free(&ptype) );
    }

    void exchangeParticles(const DomainInfo &domain, MPI_Comm comm, std::vector<Particle> &parts)
    {
        auto splitter = [domain, comm](const int dims[3], const std::vector<Particle> &parts,
                                       std::vector<std::vector<Particle>> &sendBufs) {
            for (auto& p : parts) {
                int3 procId3 = make_int3(floorf(p.r / domain.localSize));

                if (procId3.x >= dims[0] || procId3.y >= dims[1] || procId3.z >= dims[2])
                    continue;

                int procId;
                MPI_Check( MPI_Cart_rank(comm, (int*)&procId3, &procId) );
                sendBufs[procId].push_back(p);
            }
        };
        
        exchangeParticles(domain, comm, parts, splitter);
    }
    
    void exchangeParticlesChunks(const DomainInfo &domain, MPI_Comm comm, std::vector<Particle> &parts, int chunk_size)
    {
        auto splitter = [domain, comm, chunk_size](const int dims[3], const std::vector<Particle> &parts,
                                                   std::vector<std::vector<Particle>> &sendBufs) {
            for (int i = 0, k = 0; i < parts.size() / chunk_size; ++i) {
                auto com = make_float3(0);
                for (int j = 0; j < chunk_size; ++j, ++k)
                    com += parts[k].r;

                com = com / chunk_size;

                int3 procId3 = make_int3(floorf(com / domain.localSize));

                if (procId3.x >= dims[0] || procId3.y >= dims[1] || procId3.z >= dims[2])
                    continue;

                int procId;
                MPI_Check( MPI_Cart_rank(comm, (int*)&procId3, &procId) );
                sendBufs[procId].insert(sendBufs[procId].end(),
                                        parts.begin() + chunk_size * i,
                                        parts.begin() + chunk_size * (i + 1));
            }
        };
        
        exchangeParticles(domain, comm, parts, splitter);
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
