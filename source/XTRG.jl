module XTRG
export XTRG_update, XTRG_algorithm
using LinearAlgebra
include("../source/contractions.jl")
include("../source/MPO.jl")
import .contractions: contract, tensor_svd, updateLeftEnv
import .MPO: zero_mpo, add_mpo, square_mpo, trace_mpo, leftcanonicalmpo!, normalize_mpo!

"""
    XTRG_update(rho::Vector, beta::Float64; square::Bool, Nsweeps::Int, convergence::Float64, alpha::Float64, tolerance::Float64, Dmax::Int)

Function performing a single update of the XTRG algorithm from inverse temperatures beta ---> 2*beta

Parameters:
- `rho::Vector{<:AbstractArray{<:Number, 4}}`: Vector of MPO tensors corresponding to the unnormalized quantum state at inverse temperature beta.
- `beta::Float64`: Current inverse temperature of the input state rho.
- `square::Bool`: If true the updated rho is initialized as the square of rho, otherwise increase bond dimension by padding with zeros.
- `Nsweeps::Int`: Number of sweeps performed in the variational DMRG-type optimization along one direction of the chain.
- `convergence::Float64`: Threshold value for which the locally optimized tensor rho2 is assumed converged.
- `alpha::Float64`: Multiplication factor for the increment of the maximal bond dimension at each XTRG update.
- `Dmax::Int`: Total maximal bond dimension cutoff of the updated state across all temperatures.
- `tolerance::Float64`: Minimum magnitude of singular values to keep in the SVD during optimization.

Returns:
- `rho2::Vector{<:AbstractArray{<:Number, 4}}`: Vector of MPO tensors corresponding to the unnormalized quantum state at inverse temperature 2*beta.
- `beta::Float64`: Increased inverse temperature 2*beta for the updated state rho2.
- `Z::Float64`: Partition function corresponding to the updated state rho2.

"""
function XTRG_update(rho::Vector, beta::Float64; square::Bool=true, Nsweeps::Int=5, convergence::Float64=1e-8, 
    alpha::Float64=1.5, tolerance::Float64=1e-8, Dmax::Int=75)

    # Extract chain length
    L = length(rho)

    # Maximal bond dimension increased by factor alpha
    D = max(maximum([size(tensor, 1) for tensor in rho]), maximum([size(tensor, 3) for tensor in rho]))
    Nkeep = min(Int(round(alpha * D)), Dmax)

    # # # Choose initialization mode # # #
    if square == true
        rho2 = square_mpo(rho)
    else
        rho2 = deepcopy(rho)
        for _ in 1:Nkeep
            rho2 = add_mpo(rho2, zero_mpo(L))
        end
    end

    # Canonicalize the initial state
    leftcanonicalmpo!(rho2)
    
    # # # Variational DMRG-type sweeping # # #

    println("# # # Started Variational Optimization # # #")
    println("============================================")
    println("Temperature update: $beta ---> $(2*beta)")
    println("============================================")
    println(">>> # of sites = $L")
    println(">>> square mode = $square")
    println(">>> # of sweeps = $Nsweeps x 2")
    println(">>> convergence = $convergence")
    println(">>> Dmax = $Dmax")
    println(">>> alpha = $alpha")
    println(">>> Nkeep = $Nkeep")
    println(">>> tolerance = $tolerance")
    flush(stdout)

    # Prepare storage for left and right environments 
    Vlr = Vector{Array{ComplexF64, 3}}(undef, L+2)
    Vlr[1] = reshape([1], 1, 1, 1)
    Vlr[end] = reshape([1], 1, 1, 1)

    # Compute all left environments
    for itL in 1:L
        Vlr[itL+1] = updateLeftEnv(Vlr[itL], rho[itL], rho[itL], permutedims(conj(rho2[itL]), (1,4,3,2)))
    end

    for itS in 1:Nsweeps
         
        # sweeping: right ---> left
        for itL = L:-1:2

            leftEnv = contract(Vlr[itL-1], [3], rho[itL-1], [1])
            leftEnv = contract(leftEnv, [2,3], rho[itL-1], [1,4])

            rightEnv = contract(Vlr[itL+2], [3], rho[itL], [3])
            rightEnv = contract(rightEnv, [2,4], rho[itL], [3,4])

            rho_update = contract(leftEnv, [2,5], rightEnv, [2,4], [1,3,6,4,2,5]) 

            # Update two-site tensor via tensor SVD
            U, S, Vd, _ = tensor_svd(rho_update, [1,2,5]; Nkeep = Nkeep, tolerance = tolerance)
            rho2[itL] = Vd 
            rho2[itL-1] = permutedims(contract(U, [4], Diagonal(S), [1]), (1,2,4,3))

            # Update right environment for next site
            Vlr[itL+1] = updateLeftEnv(Vlr[itL+2], permutedims(rho[itL], (3,2,1,4)),
                            permutedims(rho[itL], (3,2,1,4)), permutedims(conj(rho2[itL]), (3,4,1,2)))
        end 

        # Display information of the right-left sweep
        println("Completed right-left sweep $itS / $Nsweeps")
        flush(stdout)

        # sweeping: left ---> right
        for itL = 1:(L-1)

            leftEnv = contract(Vlr[itL], [3], rho[itL], [1])
            leftEnv = contract(leftEnv, [2,3], rho[itL], [1,4])

            rightEnv = contract(Vlr[itL+3], [3], rho[itL+1], [3])
            rightEnv = contract(rightEnv, [2,4], rho[itL+1], [3,4])

            rho_update = contract(leftEnv, [2,5], rightEnv, [2,4], [1,3,6,4,2,5]) 

            # Update two-site tensor via tensor SVD
            U, S, Vd, _ = tensor_svd(rho_update, [1,2,5]; Nkeep = Nkeep, tolerance = tolerance)
            rho2[itL] = permutedims(U, (1,2,4,3)) 
            rho2[itL+1] = contract(Diagonal(S), [2], Vd, [1])

            # Update left environment for next site
            Vlr[itL+1] = updateLeftEnv(Vlr[itL], rho[itL], rho[itL], permutedims(conj(rho2[itL]), (1,4,3,2)))
        end

        # Display information of the left-right sweep
        println("Completed left-right sweep $itS / $Nsweeps")
        flush(stdout)

    end

    # Evaluate whether the optimization has sufficiently converged
    square_rho = square_mpo(rho)
    square_rho[1] = -square_rho[1]
    norm = normalize_mpo!(add_mpo(square_rho, rho2))
    println("Convergence successful: $((norm^2) < convergence)")
    println("\n")
    flush(stdout)

    # Compute the partition function 
    Z = real(trace_mpo(rho2))

    # Double the inverse temperature
    beta += beta

    return rho2, beta, Z

end


"""
    XTRG_algorithm(beta0::Float64, Nsteps::Int, rho0::Vector{<:AbstractArray{<:Number, 4}})

Function executing the XTRG algorithm to simulate the XY-Hamiltonian for a one-dimensional spin-1/2 system over a given temperature range.

Parameters:
- `beta0::Float64`: Initial inverse temperature to start the XTRG algorithm.
- `Nsteps::Int`: Number of executions of the XTRG update.
- `rho0::Vector{<:AbstractArray{<:Number, 4}}`: Vector of local MPO tensors of the initial unnormalized quantum state.

Returns:
- `betas::Vector{Float64}`: Vector of inverse temperatures at each step.
- `Zs::Vector{Float64}`: Vector of partition functions corresponding to each thermal state at the respective inverse temperatures.
- `rhos::Vector{<:AbstractArray{<:Number, 4}}`: Vector of local MPO tensors of the different thermal states.

"""
function XTRG_algorithm(beta0::Float64, Nsteps::Int, rho0::Vector)

    # Set initial values for iterative update
    beta = beta0
    rho = rho0
    Z0 = real(trace_mpo(rho0))

    # Initialize storage arrays
    betas = Vector{Float64}(undef, Nsteps + 1)
    Zs = Vector{Float64}(undef, Nsteps + 1)
    rhos = Vector{Any}(undef, Nsteps + 1)

    # Store initial values
    betas[1] = beta0
    Zs[1] = Z0
    rhos[1] = deepcopy(rho0)

    for n in 1:Nsteps

        rho, beta, Z = XTRG_update(rho, beta)

        # Store updated values
        betas[n+1] = beta
        Zs[n+1] = Z
        rhos[n+1] = deepcopy(rho)

    end

    return betas, Zs, rhos
end

end