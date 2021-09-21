#Load packages
using Parameters, Plots, DiscreteMarkovChains

@with_kw struct Primitives
    # Preferences 
    β::Float64 = 0.9932 #discount rate
    # δ::Float64 = 0.025 #depreciation rate
    α::Float64 = 1.5 #coeficient of relative risk aversion
    utility::Function = (c, v_next) -> (c^(1-α) - 1)/(1-α) + β*v_next

    # Parameters regarding the asset holding grid
    A_min::Float64 = -2 #asset holding lower bound
    A_max::Float64 =  5 #asset holding bound
    # TODO: Implement a smart way to create a better grid taking concavity in to account
    nA::Int64 = 1000 #number of capital grid points
    A_grid::Array{Float64,1} = collect(range(A_min, length = nA, stop = A_max)) #capital grid
    
    # Parameters regarding the stochastic shock
    S_vals::Array{Float64,2} = [1 0.5] #values of earning shocks
    nS::Int64 = length(S_vals) #number of earning shocks
    Π::Array{Float64,2} = [0.97  0.03; 0.5  0.5]; #transition matrix
    
    # Parameters to generate the initial distribution
    a_high = 30.0; a_low = -2.0;
end

#structure that holds model results
mutable struct Results
    val_func::Array{Float64, 2} #value function
    pol_func::Array{Float64, 2} #policy function
    μ::Array{Float64, 2} # distribution of agents in the economy
    q::Float64 # Price    
end

# Function that initializes the model
function Initialize()
    prim = Primitives() #initialize primtiives
    val_func = zeros(prim.nA, prim.nS) #initial value function guess
    pol_func = zeros(prim.nA, prim.nS) #initial policy function guess
    # temp = (prim.A_grid .- prim.a_low) ./ (prim.a_high - prim.a_low)    
    # chain = DiscreteMarkovChain(prim.Π)
    # Π_star = stationary_distribution(chain)
    # μ₀ = hcat(temp .* Π_star[1], temp .* Π_star[2])

    # ! I'm starting with a uniform distribution for now

    μ₀ = ones(prim.nA, prim.nS) / (prim.nA)
    q = (prim.β+1)/2    
    res = Results(val_func, pol_func, μ₀, q) #initialize results struct
    return prim, res #return deliverables
end

# T operator that computes the value function and update the policy function
function T(prim::Primitives, res::Results)
    @unpack val_func, q = res #unpack value function
    # TODO: Figure out why @unpack is not working 
    # @unpack k_grid, β, α, nA, S_vals, Π, nS utility = prim #unpack model primitives
    A_grid = prim.A_grid
    # β = prim.β
    # α = prim.α
    nA = prim.nA
    S_vals = prim.S_vals
    Π = prim.Π
    nS = prim.nS
    utility = prim.utility
    v_next = zeros(nA, nS) #initialize value function for next period
    # First we iterate over all posible shocks
    for s_index in 1:nS
        s = S_vals[s_index] #get current shock value
        # Then we iterate over all possible asset holdings levels
        for a_index in 1:nA
            a = A_grid[a_index] #get current asset holding level
            candidate_max = -Inf #initialize candidate max to a bad value
            # Nex we iterate over all possible future asset holdings levels
            for an_index in 1:nA
                # ! [TESTING] Use Dean's trick to speed up the computation
                a_next = A_grid[an_index] #get current future asset holding level
                c = s + a - q*a_next #get consumption
                # We check if consumption is positive, if not we skip this iteration
                if c < 0 
                    continue
                end 
                # Now we need to compute the expected value of the value function for this particular level of future asset holding
                exp_v = val_func[an_index, :]' * Π[s_index, :]
                # Now we compute the present expected value 
                val = utility(c, exp_v)
                if val > candidate_max #if the value is better than the current candidate max
                    candidate_max = val #update candidate max
                    res.pol_func[a_index, s_index] = a_next #update policy function
                elseif val < candidate_max # ! Untested: Using convexity of the value function to speed-up
                    break
                end # end if
        end #end of a' loop
        # Now we update the value function
        v_next[a_index, s_index] = candidate_max
        end #end of asset holding loop
    end # end of shock loop
    return v_next
end #end of T operator

#Value function iteration for T operator
function TV_iterate(prim::Primitives, res::Results; tol::Float64 = 1e-4, err::Float64 = 100.0)
    n = 0 #counter
    print("Iterating T operator")
    while  (err > tol) & (n < 4000)#begin iteration
        if n%10 == 0
            print("--$n")
        end
        v_next = T(prim, res) #spit out new vectors
        err = abs.(maximum(v_next.-res.val_func)) #reset error level
        res.val_func = copy(v_next) #update value function
        n+=1
        if n%500 == 0
			println("T operator ------ Iter: $n, error=$err")
		end #if
    end
    println("Value function converged in $n iterations.")
end 


function T_star(μ, pol_func, Π, A_grid, nA, nS)
    # * Seems to be working now
    μ₁ = zeros(nA, nS) #initialize next distribution 

    selected = [[], []]
    
    for s_index in 1:nS, a_index in 1:nA

        # For any combination of (s, a) we need to find 
        # how an agent can get there from previous period

        a = A_grid[a_index]

        # One way to get there is to save g^{-1}(a,s) given that previous sate is s
        # The other way is to save g^{-1}(a,s') given that previous state was s'
        # Here we just need s_1 and s_2
        g_inv_s1 = findall( pol_func[:, 1] .== a )
        g_inv_s2 = findall( pol_func[:, 2] .== a )
        
        # Now we need to find the measure of the set g^{-1}(a,s) for both values of s
        measure_g_inv_1 = sum(μ[g_inv_s1 ,1])
        measure_g_inv_2 = sum(μ[g_inv_s2 ,2])

        # Finally we need to weight that mass by the probability 
        # of actually getting there from previous state 
        # and update the new mass we will assign to that Apply_policy_function_to_μ

        μ₁[a_index, s_index] = Π[1, s_index]*measure_g_inv_1 + Π[2, s_index]*measure_g_inv_2
    end
    return μ₁
end # end of T_star


function TV_iterate_star(prim::Primitives, res::Results; tol::Float64 = 1e-4, err::Float64 = 100.0)

    @unpack val_func, pol_func, μ = res #unpack value function, policy function, and distribution
    # TODO: Figure out why @unpack is not working 
    # @unpack k_grid, β, α, nA, S_vals, Π, nS utility = prim #unpack model primitives
    A_grid = prim.A_grid
    nA = prim.nA
    Π = prim.Π
    nS = prim.nS
    print("Iterating T* operator")
    n = 1 #counter
    while  (err > tol) & (n < 4000)#begin iteration
        if n%10 == 0
            print("--$n")
        end
        μ_next = T_star(μ, pol_func, Π, A_grid, nA, nS) #spit out new vectors
        err = abs.( maximum(μ_next .- μ) ) #reset error level
        n+=1
        if err < tol
			println("T* operator ----- Iter: $n, error=$err, tolerace = $tol")
		end #if
        μ = copy(μ_next) #update distribution
    end #end of while loop

    println("T* operator converged in ", n, " iterations.")
    res.μ = μ #update distribution
end #end of TV_iterate_star

# Funtion that thest of markets clear at current prices
function solve_the_model(prim::Primitives, res::Results, tol::Float64 = 1e-4)

    @unpack μ, pol_func = res #unpack value function, policy function, and distribution
    @unpack A_grid = prim #unpack model primitives

    # Find the demand/suppy for assets

    adjusmet_step = 0.001 * res.q
    n_iter = 0
    # Find the equilibrium price
    while n_iter < 3
        # Value function iteration T operator
        TV_iterate(prim, res)

        # Value function iteration T* operator
        TV_iterate_star(prim, res)

        # excess_demand = sum( μ[:, 1] .* A_grid + μ[:, 1] .* A_grid )
        excess_demand = sum( μ .* pol_func )

        if abs(excess_demand ) < tol
            println("Markets Clear")
            break        
        elseif excess_demand  > 0
            println("Excess demand = $excess_demand > 0")
            println("Adjusting Price:\t q_old=$(res.q) ----> q_new=$(res.q + adjusmet_step)")
            res.q += adjusmet_step        
        else
            println("Excess demand = $excess_demand < 0")
            println("Adjusting Price:\t q_old=$(res.q) ----> q_new=$(res.q - adjusmet_step)")
            res.q -= adjusmet_step
        end #end of if

    end #end of while loop
end # end of market_clearing

# prim, res = Initialize();
# solve_the_model(prim, res)


# prim, res_back = Initialize();
# prim, res = Initialize();

# TV_iterate(prim, res)
# TV_iterate_star(prim, res)
# # res.μ = copy(res_back.μ)

# plot(prim.A_grid, res.val_func)
# plot(prim.A_grid, res.pol_func)


# TV_iterate_star(prim, res)


# cutoff = maximum(vcat(findall(res.μ[:,1] .!= 0), findall(res.μ[:,2] .!= 0)))
# theme(:juno)
# plot(prim.A_grid[1:cutoff], res.μ[1:cutoff,:])
# plot(prim.A_grid, cumsum(res.μ, dims=1))

# Function that solve the model

# function solve_the_model()

#     # Select an initial price between 0 and 1
#     q = 0.5

#     convergence = true # Indicator of converge 
#     while (convergence) && (iter < maxiter)
        
#         # Some fancy computation
#         # Obtain the decition rule
#         # Find the invariant distribution
#         # Check if market clears
        
#         excess_demand = 0;
#         if excess_demand > 0
#             q = 0.5*(1-q) + q
#         elseif excess_demand < 0
#             q = 0.5*q
#         else 
#             convergence = false
#         end

#     end

# end # End of function solve_the_model