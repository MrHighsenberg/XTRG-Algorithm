module XTRG
export XTRG_algorithm

"""
    XTRG_step(rho::Vector, beta::Float64, mode::Bool, Nsweeps::Int, tolerance:Float64)

Function performing a single update of the XTRG algorithm from inverse temperatures beta ---> 2*beta

Input:
- `rho::Vector`: List of the (canonicalized) local tensors of the MPO corresponding to the quantum state at inverse temperature beta.
- `beta::Float64`: Current inverse temperature of the input state rho.
- `square::Bool`: If boolean is true initialize the updated rho as the square of rho
- `Nsweeps::Int`: Number of sweeps performed in the variational DMRG-type optimization along one direction of the chain.
- `tolerance:Float64`: Threshold value for which the locally optimized tensor rho_new is assumed converged.

Output:
- `rho_new::Vector`: List of (canonicalized) local tensors of the MPO corresponding to the quantum state at inverse temperature 2*beta.
- `beta::Float64`: Increased inverse temperature 2*beta for the state rho_new. 
"""
function XTRG_step(rho::Vector, beta::Float64, square::Bool, Nsweeps::Int=5, tolerance::Float64=1e-10)

    beta += beta

    # Choose initialization mode
    if square == true
        rho_init = square_mpo(rho)
    else
        rho_init = rho
    end

    rho2 = deepcopy(rho_init)
    # Here comes the variational DMRG-type sweeping using the above methods modifying rho_new in place
    # 
    # 
    # 
    # 

    # Evaluate whether the optimization has sufficiently converged
    rho2_diff = add_mpo(rho2, [-rho2_new[1], rho2_new[2:end]])

    return rho2_new, beta

end


"""
    ..... = XTRG_algorithm(....)

Parameters:
- ...
- ...
Returns:
- `rho::Vector{Array{ComplexF64, 4}}`: the mpo representing the unnormalized thermal state

"""
function XTRG_algorithm(beta0::Float64=1e-6, betamax::Float64=1e+14, rho0::Vector)

end