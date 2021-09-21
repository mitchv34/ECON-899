using Plots, DelimitedFiles, BenchmarkTools
using DataFrames, CSV
theme(:juno)
# pgfplotsx()

cd("PS2/Solution/Fortran")

mutable struct Results_Fortran
    gid_A  :: Array{Float64, 2} # Grid 
    val_func::Array{Float64, 2} #value function
    pol_func::Array{Float64, 2} #policy function
    consumption::Array{Float64, 2} #consumption
end

# Non-Paralelized version
function run_Fortran()
    # Compile Fortran code
    run(`gfortran -fopenmp -O2 -o T_op T_operator.f90`)
    run(`./T_op`)
    results_raw =  readdlm("value_funct.csv");
    val_func = hcat(results_raw[1:1000,end], results_raw[1001:end,end])
    pol_func = hcat(results_raw[1:1000,end-1], results_raw[1001:end,end-1])
    consumption = hcat(results_raw[1:1000,end-2], results_raw[1001:end,end-2])
    grid_A = hcat(results_raw[1:1000,end-3], results_raw[1001:end,end-3])

    return Results_Fortran(grid_A, val_func, pol_func, consumption)

end # run_Fortran()


results_raw =  readdlm("value_funct.csv");
val_func = hcat(results_raw[1:1000,end], results_raw[1001:end,end])
pol_func = hcat(results_raw[1:1000,end-1], results_raw[1001:end,end-1])
consumption = hcat(results_raw[1:1000,end-2], results_raw[1001:end,end-2])
grid_A = hcat(results_raw[1:1000,end-3], results_raw[1001:end,end-3])

results_fortran = run_Fortran();


p1 = plot(results_fortran.gid_A[:,1], results_fortran.val_func[:,1], legend = :bottomright,
    xlab="a", ylab="v(a, s)", label="s = e", main="Value Function Fortran")
plot!(results_fortran.gid_A[:,1], results_fortran.val_func[:,2], 
        label="s = u")


function run_julia()
    import("")
end

include("../Julia/Hugget_model.jl")

prim, res = Initialize();
res = Results(results_fortran.val_func, results_fortran.pol_func, res.μ)

plot(res.val_func)
res.pol_func

p2 = plot(prim.A_grid, res.val_func[:,1], legend = :bottomright,
xlab="a", ylab="v(a, s)", label="s = e", main="Value Function Julia")
plot!(prim.A_grid[:,1], res.val_func[:,2], 
        label="s = u")


@time TV_iterate_star(prim, res)


μ_next = T_star(μ_next, res.pol_func, prim.Π, prim.A_grid, prim.nA, prim.nS)

plot( cumsum(μ_next[:, 1]) )

plot( cumsum(res.μ[:, 1]) )


res.pol_func