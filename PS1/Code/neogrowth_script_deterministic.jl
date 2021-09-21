# ------------------------------------------------------------------------------
# Author: Philip Coyle
# Date Created: 09/07/2021
# neogrowth_script_deterministic.jl
# ------------------------------------------------------------------------------

using Parameters, Plots
dir = "/Users/philipcoyle/Documents/School/University_of_Wisconsin/ThirdYear/Fall_2021/TA - Computation/ProblemSets/PS1/Julia"
cd(dir)
include("neogrowth_model_deterministic.jl")

## Main Code
@elapsed P, G, PFs = solve_model()
plot_pfs(dir, P, G, PFs)
