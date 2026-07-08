# Toolchain environment for building/running Mirheo on this machine (user-space).
# Usage: source .claude-env.sh
export MAMBA_ROOT_PREFIX="$HOME/micromamba"
export MIR_PREFIX="$HOME/micromamba/envs/mirheo"
export PATH="$MIR_PREFIX/bin:/usr/local/bin:/usr/bin:/bin"
export CC=/usr/bin/gcc-12
export CXX=/usr/bin/g++-12
export CUDAHOSTCXX=/usr/bin/g++-12
export HDF5_ROOT="$MIR_PREFIX"
export LD_LIBRARY_PATH="$MIR_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
# OpenMPI on a single workstation
export OMPI_MCA_btl_base_warn_component_unused=0
export OMPI_MCA_rmaps_base_oversubscribe=1
export PRTE_MCA_rmaps_default_mapping_policy=:oversubscribe
