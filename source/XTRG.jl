module XTRG
export XTRG_update, XTRG_algorithm
using LinearAlgebra
include("../source/contractions.jl")
import .contractions: contract, tensor_svd, updateLeftEnv
import .MPO: zero_mpo, add_mpo, square_mpo, trace_mpo, leftcanonicalmpo!, normalize_mpo!

"""
    XTRG_update(rho::Vector{<:AbstractArray{<:Number, 4}}, beta::Float64, mode::Bool, Nsweeps::Int, tolerance:Float64)

Function performing a single update of the XTRG algorithm from inverse temperatures beta ---> 2*beta

Parameters:
- `rho::Vector{<:AbstractArray{<:Number, 4}}`: List of MPO tensors corresponding to the unnormalized quantum state at inverse temperature beta.
- `beta::Float64`: Current inverse temperature of the input state rho.
- `square::Bool`: If true the updated rho is initialized as the square of rho, otherwise increase bond dimension by padding with zeros.
- `Nsweeps::Int`: Number of sweeps performed in the variational DMRG-type optimization along one direction of the chain.
- `convergence::Float64`: Threshold value for which the locally optimized tensor rho2 is assumed converged.
- `alpha::Float64`: Multiplication factor for the increment of the maximal bond dimension at each XTRG update.
- `Dmax::Int`: Total maximal bond dimension cutoff of the updated state across all temperatures.
- `tolerance::Float64`: Minimum magnitude of singular values to keep in the SVD during optimization.

Returns:
- `rho2::Vector{<:AbstractArray{<:Number, 4}}`: List of MPO tensors corresponding to the unnormalized quantum state at inverse temperature 2*beta.
- `beta::Float64`: Increased inverse temperature 2*beta for the state rho_new.
- `Z::Float64`

"""
function XTRG_update(rho::Vector, beta::Float64, square::Bool, Nsweeps::Int=5, convergence::Float64=1e-10, 
    alpha::Float64=1.1, tolerance::Float64=1e-12, Dmax::Int=100)

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

    print("# # # Started Variational Optimization # # #\n")
    print("Temperature update: $beta ---> $(2*beta)")
    print("""# of sites = $L | square mode = $square | # of sweeps = $Nsweeps x 2\n | convergence = $convergence
         | Nkeep = $Nkeep (Dmax = $Dmax, alpha = $alpha) | tolerance = $tolerance""")

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
            rho2[itL-1] = permutedims(U*Diagonal(S), (1,2,4,3))

            # Update right environment for next site
            Vlr[itL+1] = updateLeftEnv(permutedims(Vlr[itL+2], (3,2,1,4)), permutedims(rho[itL], (3,2,1,4)),
                            permutedims(rho[itL], (3,2,1,4)), permutedims(conj(rho2[itL]), (3,4,1,2)))

        end 

        # Display information of the right-left sweep
        print("Completed right-left sweep $itS / $Nsweeps")

        # sweeping: left ---> right
        for itL = 1:(L-1)

            leftEnv = contract(Vlr[itL], [3], rho[itL], [1])
            leftEnv = contract(leftEnv, [2,3], rho[itL], [1,4])

            rightEnv = contract(Vlr[itL+3], [3], rho[itL+1], [3])
            rightEnv = contact(rightEnv, [2,4], rho[itL+1], [3,4])

            rho_update = contract(leftEnv, [2,5], rightEnv, [2,4], [1,3,6,4,2,5]) 

            # Update two-site tensor via tensor SVD
            U, S, Vd, _ = tensor_svd(rho_update, [1,2,5]; Nkeep = Nkeep, tolerance = tolerance)
            rho2[itL] = permutedims(U, (1,2,4,3)) 
            rho2[itL+1] = Diagonal(S)*Vd

            # Update left environment for next site
            Vlr[itL+1] = updateLeftEnv(Vlr[itL+2], rho[itL], rho[itL], permutedims(conj(rho2[itL]), (1,4,3,2)))

        end

        # Display information of the left-right sweep
        print("Completed left-right sweep $itS / $Nsweeps")

    end

    # Evaluate whether the optimization has sufficiently converged
    norm = normalize_mpo!(add_mpo(square_mpo(rho), [-rho2[1], rho2[2:end]]))
    println("Convergence successful: $((norm^2) < convergence)")

    # Compute the partition function 
    Z = trace_mpo(rho2)
    @assert abs(imag(Z)) < 1e-12 "Partition function has significant imaginary part: $(imag(Z))"
    Z = real(Z)

    # Double the inverse temperature
    beta += beta

    return rho2, beta, Z

end


"""
    XTRG_algorithm(L::Int, beta0::Float64, betamax::Float64)

Function executing the XTRG algorithm to simulate the XY-Hamiltonian for a spin-1/2 system over a given temperature range.

Parameters:
- `L::Int`: length of the one-dimensional spin-1/2 system.
- `beta0::Float64`: initial inverse temperature to start the XTRG algorithm.
- `betamax::Float64`: maximal inverse temperature at which the XTRG algorithm is stopped.

Returns:
- `rhos::Dict{Float64, Vector{Array{ComplexF64, 4}}}`: The MPOs for different thermal states

"""
function XTRG_algorithm(beta0::Float64=1e-6, betamax::Float64=1e+14, rho0::Vector)

end

end