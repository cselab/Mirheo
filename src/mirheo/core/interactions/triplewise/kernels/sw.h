// Copyright 2020 ETH Zurich. All Rights Reserved.

// Code: Noah Baumann Bachelor Thesis
// SW: Stillinger-Weber potiential
// 3Body SW potential

#pragma once

//maybe fix
#include <mirheo/core/interactions/pairwise/kernels/fetchers.h>

#include "interface.h"
#include "parameters.h"

#include <mirheo/core/utils/cuda_common.h>
#include <array>

namespace mirheo
{

class SW3Handler : public ParticleFetcher
{
public:
    using ViewType = PVview;
    using ParticleType = Particle;

    SW3Handler(real rc, real lambda, real epsilon, real theta, real gamma, real sigma) : 
        ParticleFetcher(rc),
        lambda_epsilon_(lambda*epsilon),
        cos_theta_(math::cos(theta)),   //theta = 1.910633236 -> cos(theta) ~ -1/3
        gamma_sigma_(gamma*sigma)
        {}

    /*
    __D__ inline std::array<real3, 3> operator()(ParticleType p_i, ParticleType p_j, ParticleType p_k, const bool interacting_ij, const bool interacting_jk, const bool interacting_ki) const
    {
        const real3 r_ij = p_i.r - p_j.r;
        const real3 r_jk = p_j.r - p_k.r;
        const real3 r_ki = p_k.r - p_i.r;

        const real dr_ij2 = dot(r_ij, r_ij);
        const real dr_jk2 = dot(r_jk, r_jk);
        const real dr_ki2 = dot(r_ki, r_ki);

        const real dr_ij = math::sqrt(dr_ij2);
        const real dr_jk = math::sqrt(dr_jk2);
        const real dr_ki = math::sqrt(dr_ki2);

        const real3 r_ij_hat = r_ij/dr_ij;
        const real3 r_jk_hat = r_jk/dr_jk;
        const real3 r_ki_hat = r_ki/dr_ki;

        const real cos_theta_jik = -dot(r_ij_hat, r_ki_hat);
        const real cos_theta_ijk = -dot(r_ij_hat, r_jk_hat);
        const real cos_theta_ikj = -dot(r_ki_hat, r_jk_hat);

        const real dr_ij_inv = 1.0_r/dr_ij;
        const real dr_jk_inv = 1.0_r/dr_jk;
        const real dr_ki_inv = 1.0_r/dr_ki;

        const real dr_ij_rc_inv = 1.0_r/(dr_ij-rc_);
        const real dr_jk_rc_inv = 1.0_r/(dr_jk-rc_);
        const real dr_ki_rc_inv = 1.0_r/(dr_ki-rc_);

        const real3 zeros = make_real3(0.0_r);        
        real3 out[3] = {zeros, zeros, zeros};

        //h_jik
        if (interacting_ij && interacting_ki) {
            const real cos_cos = (cos_theta_jik - cos_theta_);

            const real exp = math::exp(gamma_sigma_*dr_ij_rc_inv + gamma_sigma_*dr_ki_rc_inv);

            const real exp_lambda_epsilon_cos_cos = exp*lambda_epsilon_*cos_cos;

            const real3 h_jik_j = exp_lambda_epsilon_cos_cos*(2.0_r*( r_ki_hat*dr_ij_inv + cos_theta_jik*r_ij_hat*dr_ij_inv) + cos_cos*gamma_sigma_*dr_ij_rc_inv*dr_ij_rc_inv*r_ij_hat);
            const real3 h_jik_k = exp_lambda_epsilon_cos_cos*(2.0_r*(-r_ij_hat*dr_ki_inv - cos_theta_jik*r_ki_hat*dr_ki_inv) - cos_cos*gamma_sigma_*dr_ki_rc_inv*dr_ki_rc_inv*r_ki_hat);
            const real3 h_jik_i = -h_jik_j - h_jik_k;
            out[0] -= h_jik_i;
            out[1] -= h_jik_j;
            out[2] -= h_jik_k;
        }

        //h_ijk
        if (interacting_ij && interacting_jk) {
            const real cos_cos = (cos_theta_ijk - cos_theta_);

            const real exp = math::exp(gamma_sigma_*dr_ij_rc_inv + gamma_sigma_*dr_jk_rc_inv);

            const real exp_lambda_epsilon_cos_cos = exp*lambda_epsilon_*cos_cos;

            const real3 h_ijk_i = exp_lambda_epsilon_cos_cos*(2.0_r*(-r_jk_hat*dr_ij_inv - cos_theta_ijk*r_ij_hat*dr_ij_inv) - cos_cos*gamma_sigma_*dr_ij_rc_inv*dr_ij_rc_inv*r_ij_hat);
            const real3 h_ijk_k = exp_lambda_epsilon_cos_cos*(2.0_r*( r_ij_hat*dr_jk_inv + cos_theta_ijk*r_jk_hat*dr_jk_inv) + cos_cos*gamma_sigma_*dr_jk_rc_inv*dr_jk_rc_inv*r_jk_hat);
            const real3 h_ijk_j = -h_ijk_i - h_ijk_k;
            out[0] -= h_ijk_i;
            out[1] -= h_ijk_j;
            out[2] -= h_ijk_k;
        }

        //h_ikj
        if (interacting_ki && interacting_jk) {
            const real cos_cos = (cos_theta_ikj - cos_theta_);

            const real exp = math::exp(gamma_sigma_*dr_ki_rc_inv + gamma_sigma_*dr_jk_rc_inv);

            const real exp_lambda_epsilon_cos_cos = exp*lambda_epsilon_*cos_cos;

            const real3 h_ikj_i = exp_lambda_epsilon_cos_cos*(2.0_r*( r_jk_hat*dr_ki_inv + cos_theta_ikj*r_ki_hat*dr_ki_inv) + cos_cos*gamma_sigma_*dr_ki_rc_inv*dr_ki_rc_inv*r_ki_hat);
            const real3 h_ikj_j = exp_lambda_epsilon_cos_cos*(2.0_r*(-r_ki_hat*dr_jk_inv - cos_theta_ikj*r_jk_hat*dr_jk_inv) - cos_cos*gamma_sigma_*dr_jk_rc_inv*dr_jk_rc_inv*r_jk_hat);
            const real3 h_ikj_k = -h_ikj_i - h_ikj_j;
            out[0] -= h_ikj_i;
            out[1] -= h_ikj_j;
            out[2] -= h_ikj_k;
        }
        return {out[0], out[1], out[2]};
    }
    */
    
    __D__ inline std::array<real3, 3> operator()(ParticleType p_i, ParticleType p_j, ParticleType p_k, const bool interacting_ij, const bool interacting_jk, const bool interacting_ki) const
    {
        //all
        if(interacting_ij && interacting_jk && interacting_ki){
            const real3 r_ij = p_i.r - p_j.r;
            const real3 r_jk = p_j.r - p_k.r;
            const real3 r_ki = p_k.r - p_i.r;

            const real dr_ij2 = dot(r_ij, r_ij);
            const real dr_jk2 = dot(r_jk, r_jk);
            const real dr_ki2 = dot(r_ki, r_ki);

            const real dr_ij = math::sqrt(dr_ij2);
            const real dr_jk = math::sqrt(dr_jk2);
            const real dr_ki = math::sqrt(dr_ki2);

            const real3 r_ij_hat = r_ij/dr_ij;
            const real3 r_jk_hat = r_jk/dr_jk;
            const real3 r_ki_hat = r_ki/dr_ki;

            const real cos_theta_jik = -dot(r_ij_hat, r_ki_hat);
            const real cos_theta_ijk = -dot(r_ij_hat, r_jk_hat);
            const real cos_theta_ikj = -dot(r_ki_hat, r_jk_hat);

            const real dr_ij_inv = 1.0_r/dr_ij;
            const real dr_jk_inv = 1.0_r/dr_jk;
            const real dr_ki_inv = 1.0_r/dr_ki;

            const real dr_ij_rc_inv = 1.0_r/(dr_ij-rc_);
            const real dr_jk_rc_inv = 1.0_r/(dr_jk-rc_);
            const real dr_ki_rc_inv = 1.0_r/(dr_ki-rc_);

            //h_jik
            real cos_cos = (cos_theta_jik - cos_theta_);

            real exp = math::exp(gamma_sigma_*dr_ij_rc_inv + gamma_sigma_*dr_ki_rc_inv);

            real exp_lambda_epsilon_cos_cos = exp*lambda_epsilon_*cos_cos;

            const real3 h_jik_j = exp_lambda_epsilon_cos_cos*(2.0_r*( r_ki_hat*dr_ij_inv + cos_theta_jik*r_ij_hat*dr_ij_inv) + cos_cos*gamma_sigma_*dr_ij_rc_inv*dr_ij_rc_inv*r_ij_hat);
            const real3 h_jik_k = exp_lambda_epsilon_cos_cos*(2.0_r*(-r_ij_hat*dr_ki_inv - cos_theta_jik*r_ki_hat*dr_ki_inv) - cos_cos*gamma_sigma_*dr_ki_rc_inv*dr_ki_rc_inv*r_ki_hat);
            const real3 h_jik_i = -h_jik_j - h_jik_k;

            //h_ijk
            cos_cos = (cos_theta_ijk - cos_theta_);

            exp = math::exp(gamma_sigma_*dr_ij_rc_inv + gamma_sigma_*dr_jk_rc_inv);

            exp_lambda_epsilon_cos_cos = exp*lambda_epsilon_*cos_cos;

            const real3 h_ijk_i = exp_lambda_epsilon_cos_cos*(2.0_r*(-r_jk_hat*dr_ij_inv - cos_theta_ijk*r_ij_hat*dr_ij_inv) - cos_cos*gamma_sigma_*dr_ij_rc_inv*dr_ij_rc_inv*r_ij_hat);
            const real3 h_ijk_k = exp_lambda_epsilon_cos_cos*(2.0_r*( r_ij_hat*dr_jk_inv + cos_theta_ijk*r_jk_hat*dr_jk_inv) + cos_cos*gamma_sigma_*dr_jk_rc_inv*dr_jk_rc_inv*r_jk_hat);
            const real3 h_ijk_j = -h_ijk_i - h_ijk_k;

            //h_ikj
            cos_cos = (cos_theta_ikj - cos_theta_);

            exp = math::exp(gamma_sigma_*dr_ki_rc_inv + gamma_sigma_*dr_jk_rc_inv);

            exp_lambda_epsilon_cos_cos = exp*lambda_epsilon_*cos_cos;

            const real3 h_ikj_i = exp_lambda_epsilon_cos_cos*(2.0_r*( r_jk_hat*dr_ki_inv + cos_theta_ikj*r_ki_hat*dr_ki_inv) + cos_cos*gamma_sigma_*dr_ki_rc_inv*dr_ki_rc_inv*r_ki_hat);
            const real3 h_ikj_j = exp_lambda_epsilon_cos_cos*(2.0_r*(-r_ki_hat*dr_jk_inv - cos_theta_ikj*r_jk_hat*dr_jk_inv) - cos_cos*gamma_sigma_*dr_jk_rc_inv*dr_jk_rc_inv*r_jk_hat);
            const real3 h_ikj_k = -h_ikj_i - h_ikj_j;

            return {-(h_jik_i + h_ijk_i + h_ikj_i), -(h_jik_j + h_ijk_j + h_ikj_j), -(h_jik_k + h_ijk_k + h_ikj_k)};
        }
        //h_ijk
        else if(interacting_ij && interacting_jk){
            const real3 r_ij = p_i.r - p_j.r;
            const real3 r_jk = p_j.r - p_k.r;
            
            const real dr_ij2 = dot(r_ij, r_ij);
            const real dr_jk2 = dot(r_jk, r_jk);
            
            const real dr_ij = math::sqrt(dr_ij2);
            const real dr_jk = math::sqrt(dr_jk2);
            
            const real3 r_ij_hat = r_ij/dr_ij;
            const real3 r_jk_hat = r_jk/dr_jk;
            
            const real dr_ij_inv = 1.0_r/dr_ij;
            const real dr_jk_inv = 1.0_r/dr_jk;
            
            const real dr_ij_rc_inv = 1.0_r/(dr_ij-rc_);
            const real dr_jk_rc_inv = 1.0_r/(dr_jk-rc_);
            
            const real cos_theta_ijk = -dot(r_ij_hat, r_jk_hat);

            //h_ijk
            const real cos_cos = (cos_theta_ijk - cos_theta_);

            const real exp = math::exp(gamma_sigma_*dr_ij_rc_inv + gamma_sigma_*dr_jk_rc_inv);

            const real exp_lambda_epsilon_cos_cos = exp*lambda_epsilon_*cos_cos;

            const real3 h_ijk_i = exp_lambda_epsilon_cos_cos*(2.0_r*(-r_jk_hat*dr_ij_inv - cos_theta_ijk*r_ij_hat*dr_ij_inv) - cos_cos*gamma_sigma_*dr_ij_rc_inv*dr_ij_rc_inv*r_ij_hat);
            const real3 h_ijk_k = exp_lambda_epsilon_cos_cos*(2.0_r*( r_ij_hat*dr_jk_inv + cos_theta_ijk*r_jk_hat*dr_jk_inv) + cos_cos*gamma_sigma_*dr_jk_rc_inv*dr_jk_rc_inv*r_jk_hat);
            const real3 h_ijk_j = -h_ijk_i - h_ijk_k;


            return {-h_ijk_i, -h_ijk_j, -h_ijk_k};
        }
        // h_jik
        else if(interacting_ij && interacting_ki){
            const real3 r_ij = p_i.r - p_j.r;
            const real3 r_ki = p_k.r - p_i.r;

            const real dr_ij2 = dot(r_ij, r_ij);
            const real dr_ki2 = dot(r_ki, r_ki);

            const real dr_ij = math::sqrt(dr_ij2);
            const real dr_ki = math::sqrt(dr_ki2);

            const real3 r_ij_hat = r_ij/dr_ij;
            const real3 r_ki_hat = r_ki/dr_ki;

            const real dr_ij_inv = 1.0_r/dr_ij;
            const real dr_ki_inv = 1.0_r/dr_ki;

            const real dr_ij_rc_inv = 1.0_r/(dr_ij-rc_);
            const real dr_ki_rc_inv = 1.0_r/(dr_ki-rc_);

            const real cos_theta_jik = -dot(r_ij_hat, r_ki_hat);

            //h_jik
            const real cos_cos = (cos_theta_jik - cos_theta_);

            const real exp = math::exp(gamma_sigma_*dr_ij_rc_inv + gamma_sigma_*dr_ki_rc_inv);

            const real exp_lambda_epsilon_cos_cos = exp*lambda_epsilon_*cos_cos;

            const real3 h_jik_j = exp_lambda_epsilon_cos_cos*(2.0_r*( r_ki_hat*dr_ij_inv + cos_theta_jik*r_ij_hat*dr_ij_inv) + cos_cos*gamma_sigma_*dr_ij_rc_inv*dr_ij_rc_inv*r_ij_hat);
            const real3 h_jik_k = exp_lambda_epsilon_cos_cos*(2.0_r*(-r_ij_hat*dr_ki_inv - cos_theta_jik*r_ki_hat*dr_ki_inv) - cos_cos*gamma_sigma_*dr_ki_rc_inv*dr_ki_rc_inv*r_ki_hat);
            const real3 h_jik_i = -h_jik_j - h_jik_k;


            return {-h_jik_i, -h_jik_j, -h_jik_k};
        }
        // h_ikj
        else if(interacting_jk && interacting_ki){
            const real3 r_jk = p_j.r - p_k.r;
            const real3 r_ki = p_k.r - p_i.r;

            const real dr_jk2 = dot(r_jk, r_jk);
            const real dr_ki2 = dot(r_ki, r_ki);

            const real dr_jk = math::sqrt(dr_jk2);
            const real dr_ki = math::sqrt(dr_ki2);

            const real3 r_jk_hat = r_jk/dr_jk;
            const real3 r_ki_hat = r_ki/dr_ki;

            const real dr_jk_inv = 1.0_r/dr_jk;
            const real dr_ki_inv = 1.0_r/dr_ki;

            const real dr_jk_rc_inv = 1.0_r/(dr_jk-rc_);
            const real dr_ki_rc_inv = 1.0_r/(dr_ki-rc_);

            const real cos_theta_ikj = -dot(r_ki_hat, r_jk_hat);


            //h_ikj
            const real cos_cos = (cos_theta_ikj - cos_theta_);

            const real exp = math::exp(gamma_sigma_*dr_ki_rc_inv + gamma_sigma_*dr_jk_rc_inv);

            const real exp_lambda_epsilon_cos_cos = exp*lambda_epsilon_*cos_cos;

            const real3 h_ikj_i = exp_lambda_epsilon_cos_cos*(2.0_r*( r_jk_hat*dr_ki_inv + cos_theta_ikj*r_ki_hat*dr_ki_inv) + cos_cos*gamma_sigma_*dr_ki_rc_inv*dr_ki_rc_inv*r_ki_hat);
            const real3 h_ikj_j = exp_lambda_epsilon_cos_cos*(2.0_r*(-r_ki_hat*dr_jk_inv - cos_theta_ikj*r_jk_hat*dr_jk_inv) - cos_cos*gamma_sigma_*dr_jk_rc_inv*dr_jk_rc_inv*r_jk_hat);
            const real3 h_ikj_k = -h_ikj_i - h_ikj_j;

            return {-h_ikj_i, -h_ikj_j, -h_ikj_k};
        }
        
        const real3 zeros = make_real3(0.0_r);        
        return {zeros, zeros, zeros};
        
    }
    
    /*
    __D__ inline std::array<real3, 3> operator()(ParticleType p_i, ParticleType p_j, ParticleType p_k, int id_i, int id_j, int id_k) const
    {
        const real3 r_ij = p_i.r - p_j.r;
        const real3 r_jk = p_j.r - p_k.r;
        const real3 r_ki = p_k.r - p_i.r;

        const real dr_ij2 = dot(r_ij, r_ij);
        const real dr_jk2 = dot(r_jk, r_jk);
        const real dr_ki2 = dot(r_ki, r_ki);

        const real dr_ij = math::sqrt(dr_ij2);
        const real dr_jk = math::sqrt(dr_jk2);
        const real dr_ki = math::sqrt(dr_ki2);

        const real3 r_ij_hat = r_ij/dr_ij;
        const real3 r_jk_hat = r_jk/dr_jk;
        const real3 r_ki_hat = r_ki/dr_ki;

        const real cos_theta_jik = -dot(r_ij_hat, r_ki_hat);
        const real cos_theta_ijk = -dot(r_ij_hat, r_jk_hat);
        const real cos_theta_ikj = -dot(r_ki_hat, r_jk_hat);

        const real dr_ij_inv = 1.0_r/dr_ij;
        const real dr_jk_inv = 1.0_r/dr_jk;
        const real dr_ki_inv = 1.0_r/dr_ki;

        const real dr_ij_rc_inv = 1.0_r/(dr_ij-rc_);
        const real dr_jk_rc_inv = 1.0_r/(dr_jk-rc_);
        const real dr_ki_rc_inv = 1.0_r/(dr_ki-rc_);

        //h_jik
        real cos_cos = (cos_theta_jik - cos_theta_);

        real exp = math::exp(gamma_sigma_*dr_ij_rc_inv + gamma_sigma_*dr_ki_rc_inv);

        real exp_lambda_epsilon_cos_cos = exp*lambda_epsilon_*cos_cos;

        const real3 h_jik_j = exp_lambda_epsilon_cos_cos*(2.0_r*( r_ki_hat*dr_ij_inv + cos_theta_jik*r_ij_hat*dr_ij_inv) + cos_cos*gamma_sigma_*dr_ij_rc_inv*dr_ij_rc_inv*r_ij_hat);
        const real3 h_jik_k = exp_lambda_epsilon_cos_cos*(2.0_r*(-r_ij_hat*dr_ki_inv - cos_theta_jik*r_ki_hat*dr_ki_inv) - cos_cos*gamma_sigma_*dr_ki_rc_inv*dr_ki_rc_inv*r_ki_hat);
        const real3 h_jik_i = -h_jik_j - h_jik_k;

        //h_ijk
        cos_cos = (cos_theta_ijk - cos_theta_);

        exp = math::exp(gamma_sigma_*dr_ij_rc_inv + gamma_sigma_*dr_jk_rc_inv);

        exp_lambda_epsilon_cos_cos = exp*lambda_epsilon_*cos_cos;

        const real3 h_ijk_i = exp_lambda_epsilon_cos_cos*(2.0_r*(-r_jk_hat*dr_ij_inv - cos_theta_ijk*r_ij_hat*dr_ij_inv) - cos_cos*gamma_sigma_*dr_ij_rc_inv*dr_ij_rc_inv*r_ij_hat);
        const real3 h_ijk_k = exp_lambda_epsilon_cos_cos*(2.0_r*( r_ij_hat*dr_jk_inv + cos_theta_ijk*r_jk_hat*dr_jk_inv) + cos_cos*gamma_sigma_*dr_jk_rc_inv*dr_jk_rc_inv*r_jk_hat);
        const real3 h_ijk_j = -h_ijk_i - h_ijk_k;

        //h_ikj
        cos_cos = (cos_theta_ikj - cos_theta_);

        exp = math::exp(gamma_sigma_*dr_ki_rc_inv + gamma_sigma_*dr_jk_rc_inv);

        exp_lambda_epsilon_cos_cos = exp*lambda_epsilon_*cos_cos;

        const real3 h_ikj_i = exp_lambda_epsilon_cos_cos*(2.0_r*( r_jk_hat*dr_ki_inv + cos_theta_ikj*r_ki_hat*dr_ki_inv) + cos_cos*gamma_sigma_*dr_ki_rc_inv*dr_ki_rc_inv*r_ki_hat);
        const real3 h_ikj_j = exp_lambda_epsilon_cos_cos*(2.0_r*(-r_ki_hat*dr_jk_inv - cos_theta_ikj*r_jk_hat*dr_jk_inv) - cos_cos*gamma_sigma_*dr_jk_rc_inv*dr_jk_rc_inv*r_jk_hat);
        const real3 h_ikj_k = -h_ikj_i - h_ikj_j;


        if(dr_ij2 >= rc2_){
            if(dr_jk2 < rc2_ && dr_ki2 < rc2_){     //(ij, jk, ki): (n,y,y)
                return {-h_ikj_i, -h_ikj_j, -h_ikj_k};
            }else{                                  //(ij, jk, ki): (n,y,n),(n,n,y),(n,n,n)
                const real3 zeros = make_real3(0.0_r);        
                return {zeros, zeros, zeros};          //shouldn't make it to this point
            }
        }else if(dr_jk2 >= rc2_ && dr_ki2 >= rc2_){ //(ij, jk, ki): (y,n,n)
            const real3 zeros = make_real3(0.0_r);        
            return {zeros, zeros, zeros};
        }else{                                      //(ij, jk, ki): (y,y,y),(y,y,n),(y,n,y)
            if(dr_jk2 < rc2_ && dr_ki2 < rc2_){     //(ij, jk, ki): (y,y,y)
                return {-(h_jik_i + h_ijk_i + h_ikj_i), -(h_jik_j + h_ijk_j + h_ikj_j), -(h_jik_k + h_ijk_k + h_ikj_k)};
            }else if(dr_jk2 < rc2_){                //(ij, jk, ki): (y,y,n)
                return {-h_ijk_i, -h_ijk_j, -h_ijk_k};
            }else{                                  //(ij, jk, ki): (y,n,y)
                return {-h_jik_i, -h_jik_j, -h_jik_k};
            }
        }
    }
    */

private:
    real lambda_epsilon_;
    real cos_theta_;
    real gamma_sigma_;
};

class SW3 : public TriplewiseKernelBase
{
public:
    using HandlerType = SW3Handler; ///< handler type corresponding to this object
    using ParamsType  = SW3Params; ///< parameters that are used to create this object
    using ViewType    = PVview;     ///< compatible view type

    /// Constructor
    SW3(real rc, real lambda, real epsilon, real theta, real gamma, real sigma) : handler_(rc, lambda, epsilon, theta, gamma, sigma) {}

    /// Generic constructor
    SW3(real rc, const ParamsType& p, __UNUSED long seed=42424242) :
        SW3(rc, p.lambda, p.epsilon, p.theta, p.gamma, p.sigma)
    {}

    /// get the handler that can be used on device
    const HandlerType& handler() const { return handler_; }

private:
    SW3Handler handler_;
};

} // namespace mirheo
