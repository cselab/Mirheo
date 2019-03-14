#pragma once

#include "interface.h"
#include <memory>
#include <limits>
#include <core/utils/pytypes.h>

template <class DensityKernel>
class InteractionSDPDDensity : public Interaction
{
public:
    InteractionSDPDDensity(const YmrState *state, std::string name, float rc, DensityKernel densityKernel);

    ~InteractionSDPDDensity();

    void setPrerequisites(ParticleVector *pv1, ParticleVector *pv2, CellList *cl1, CellList *cl2) override;
    
    std::vector<InteractionChannel> getIntermediateOutputChannels() const override;
    std::vector<InteractionChannel> getFinalOutputChannels() const override;
    
    void local (ParticleVector *pv1, ParticleVector *pv2, CellList *cl1, CellList *cl2, cudaStream_t stream) override;
    void halo  (ParticleVector *pv1, ParticleVector *pv2, CellList *cl1, CellList *cl2, cudaStream_t stream) override;
        
protected:

    std::unique_ptr<Interaction> impl;

    DensityKernel densityKernel;
};

class BasicInteractionSDPD : public Interaction
{
public:
    
    BasicInteractionSDPD(const YmrState *state, std::string name, float rc,
                         float viscosity, float kBT);

    ~BasicInteractionSDPD();

    void setPrerequisites(ParticleVector *pv1, ParticleVector *pv2, CellList *cl1, CellList *cl2) override;

    std::vector<InteractionChannel> getIntermediateInputChannels() const override;
    std::vector<InteractionChannel> getFinalOutputChannels() const override;
    
    void local (ParticleVector *pv1, ParticleVector *pv2, CellList *cl1, CellList *cl2, cudaStream_t stream) override;
    void halo  (ParticleVector *pv1, ParticleVector *pv2, CellList *cl1, CellList *cl2, cudaStream_t stream) override;
        
protected:

    std::unique_ptr<Interaction> impl;

    float viscosity, kBT;
};

template <class PressureEOS, class DensityKernel>
class InteractionSDPD : public BasicInteractionSDPD
{
public:
    
    InteractionSDPD(const YmrState *state, std::string name, float rc,
                    PressureEOS pressure, DensityKernel densityKernel,
                    float viscosity, float kBT);

    ~InteractionSDPD();

    void setPrerequisites(ParticleVector *pv1, ParticleVector *pv2, CellList *cl1, CellList *cl2) override;

    std::vector<InteractionChannel> getIntermediateInputChannels() const override;
    std::vector<InteractionChannel> getFinalOutputChannels() const override;
    
    void local (ParticleVector *pv1, ParticleVector *pv2, CellList *cl1, CellList *cl2, cudaStream_t stream) override;
    void halo  (ParticleVector *pv1, ParticleVector *pv2, CellList *cl1, CellList *cl2, cudaStream_t stream) override;
        
protected:

    InteractionSDPD(const YmrState *state, std::string name, float rc,
                    PressureEOS pressure, DensityKernel densityKernel,
                    float viscosity, float kBT, bool allocateImpl);
    
    std::unique_ptr<Interaction> impl;

    PressureEOS pressure;
    DensityKernel densityKernel;
};

