# ------------------------------------------------------------------------------
# Author: Philip Coyle
# neogrowth_model_deterministic.jl
# ------------------------------------------------------------------------------


## Structures
@with_kw struct Params
    β::Float64 = 0.99
    δ::Float64 = 0.025
    θ::Float64 = 0.36

    tol::Float64 = 1e-4
    maxit::Int64 = 10000
end

@with_kw struct Grids
    Zg::Float64 = 1.25
    Zb::Float64 = 0.2

    k_lb::Float64 = 0.01
    k_ub::Float64 = 45.0
    n_k::Int64 = 1000
    k_grid::Array{Float64,1} = range(k_lb, stop = k_ub, length = n_k)
end

mutable struct PolFuncs
    pf_c::Array{Float64,1}
    pf_k::Array{Float64,1}
    pf_v::Array{Float64,1}
end


## Functions
function solve_model()
    P = Params()
    G = Grids()

    @unpack n_k = G
    # Initial Guess
    pf_c = zeros(n_k)
    pf_k = zeros(n_k)
    pf_v = zeros(n_k)
    PFs = PolFuncs(pf_c, pf_k, pf_v)

    converged = 0
    it = 1
    while converged == 0 && it < P.maxit
        @unpack pf_v, pf_k, pf_c = PFs

        pf_c_up, pf_k_up, pf_v_up = Bellman(P, G, PFs)

        diff_v = maximum(abs.(pf_v_up - pf_v))
        diff_k = maximum(abs.(pf_k_up - pf_k))
        diff_c = maximum(abs.(pf_c_up - pf_c))

        max_diff = diff_v + diff_k + diff_c

        if mod(it, 50) == 0 || max_diff < P.tol
            println(" ")
            println("*************************************************")
            println("AT ITERATION = ", it)
            println("MAX DIFFERENCE = ", max_diff)
            println("*************************************************")

            if max_diff < P.tol
                converged = 1
            end
        end
        # Update the policy functions
        PFs = PolFuncs(pf_c_up, pf_k_up, pf_v_up)
        it = it + 1
    end

    return P, G, PFs
end

function Bellman(P::Params, G::Grids, PFs::PolFuncs)
    @unpack β, δ, θ = P
    @unpack n_k, k_grid = G
    @unpack pf_c, pf_k, pf_v = PFs

    # To make updating work
    pf_k_up = zeros(n_k)
    pf_c_up = zeros(n_k)
    pf_v_up = zeros(n_k)

    for (i_k, k_today) in enumerate(k_grid)
        # Must be defined outside loop.
        v_today = log(0)
        c_today = log(0)
        k_tomorrow = log(0)

        y_today = k_today^θ

        # Find optimal investment/consumption given capital level today
        for (i_kpr, k_temp) in enumerate(k_grid)
            c_temp = y_today + (1 - δ) * k_today - k_temp
            v_tomorrow = pf_v[i_kpr]
            if c_temp < 0
                v_temp = log(0) + β * v_tomorrow
            else
                v_temp = log(c_temp) + β * v_tomorrow
            end

            if v_temp > v_today
                v_today = v_temp
                c_today = c_temp
                k_tomorrow = k_temp
            end
        end

        # Update PFs
        pf_k_up[i_k] = k_tomorrow
        pf_c_up[i_k] = c_today
        pf_v_up[i_k] = v_today
    end

    return pf_c_up, pf_k_up, pf_v_up
end

function plot_pfs(dir::String, P::Params, G::Grids, PFs::PolFuncs)
    @unpack k_grid = G
    @unpack pf_c, pf_k, pf_v = PFs

    pf_1 = plot(k_grid,pf_v,title="Value Function",legend = false,color=:blue,lw = 2);

    pf_2 = plot(k_grid,pf_k,title="Capital Investment",legend = false,color=:blue, lw = 2);

    pf_3 = plot(k_grid,pf_k - k_grid, title="Net Capital Inv.",legend = false,color=:blue, lw = 2);

    pf = plot(pf_1,pf_2,pf_3,layout=(1,3),size = (600,400)) #Size can be adjusted so don't need to mess around with 'blank space'
    xlabel!("Initial Capital Stock")
end
