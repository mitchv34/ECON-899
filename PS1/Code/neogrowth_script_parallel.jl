# ------------------------------------------------------------------------------
# Author: Philip Coyle
# Date Created: 09/07/2021
# neogrowth_script_parallel.jl
# ------------------------------------------------------------------------------
using Distributed
addprocs(6)

@everywhere using Parameters, Plots, SharedArrays
@everywhere include("neogrowth_model_parallel.jl")

## Main Code
@elapsed G, PFs = solve_model()
# plot_pfs(dir, G, PFs)
