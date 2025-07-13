module XTRG
export XTRG_update, XTRG_algorithm

"""
    XTRG_update(rho::Vector{<:AbstractArray{<:Number, 4}}, beta::Float64, mode::Bool, Nsweeps::Int, tolerance:Float64)

Function performing a single update of the XTRG algorithm from inverse temperatures beta ---> 2*beta

Parameters:
- `rho::Vector{<:AbstractArray{<:Number, 4}}`: List of the (canonicalized) local tensors of the MPO corresponding to the quantum state at inverse temperature beta.
- `beta::Float64`: Current inverse temperature of the input state rho.
- `square::Bool`: If boolean is true initialize the updated rho as the square of rho
- `Nsweeps::Int`: Number of sweeps performed in the variational DMRG-type optimization along one direction of the chain.
- `tolerance::Float64`: Threshold value for which the locally optimized tensor rho_new is assumed converged.

Returns:
- `rho_new::Vector{<:AbstractArray{<:Number, 4}}`: List of (canonicalized) local tensors of the MPO corresponding to the quantum state at inverse temperature 2*beta.
- `beta::Float64`: Increased inverse temperature 2*beta for the state rho_new. 
"""
function XTRG_update(rho::Vector, beta::Float64, square::Bool, Nsweeps::Int=5, tolerance::Float64=1e-10)

    # Double the inverse temperature
    beta += beta

    # Choose initialization mode
    if square == true
        rho_init = square_mpo(rho)
    else
        rho_init = rho
    end

    rho2 = deepcopy(rho_init)
    
    # Variational DMRG-type sweeping 
    

    # Evaluate whether the optimization has sufficiently converged
    rho2_diff = add_mpo(rho2, [-rho2_new[1], rho2_new[2:end]])

    return rho2_new, beta

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