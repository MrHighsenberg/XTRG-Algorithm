module XTRG
export square_mpo, XTRG_algorithm
using LinearAlgebra
include("lecture_functions.jl")
using .lecture_functions: contract

"""
    mpo2 = square_mpo(mpo, Dmax, Nsweep)

mpo2 is an mpo representing the square of the operator represented by mpo. The dimension of each virtual bond of mpo2 is at most Dmax.

Parameters:
- `mpo::Vector{Array{ComplexF64, 4}}`: the mpo representing the operator to be squared (each tensor has legs ordered as left - bottom - right - top)
- `Dmax::Int`: max bond dimension of output mpo
- `Nsweep::Int`: Number of sweeps will be 2*Nsweep (Nsweep left -> right sweeps, Nsweep right -> left sweeps)
Returns:
- `mpo2::Vector{Array{ComplexF64, 4}}`: the mpo representing the squared operator

"""
function square_mpo(mpo, Dmax, Nsweep)
    L = length(mpo)
    d = size(mpo[1], 2) # we assume the local dimension is equal at all sites
    max_L_virt_bd = maximum([size(W, 1) for W in mpo])
    max_R_virt_bd = maximum([size(W, 3) for W in mpo])
    mpo2 = Vector{Any}(undef, L)
    D = max(max_L_virt_bd, max_R_virt_bd) # max bond dimension of W tensors that are elements of mpo. (Bond dimensions don't have to be equal)
    if D^2 <= Dmax # in this case we won't need to truncate/optimize anything
        for i in (1:L)
            W = mpo[i]
            DL = size(W, 1)
            DR = size(W, 3)
            WW =  contract(W, [4], W, [2])
            WW1 = permutedims(WW, (1,4,3,2,5,6))
            WW2 = permutedims(WW1, (1,2,4,3,5,6))
            WW3 = reshape(WW2, (DL^2, d, DR^2, d))
            mpo2[i] = WW3 # when "fusing" the two horizontal virtual legs, these are ordered bottom - top
        end
    else # in this case we need to truncate and optimize
        # mpo2 = ...
    end
    return mpo2
end

"""
    ..... = XTRG_algorithm(....)

mpo2 is an mpo representing the square of the operator represented by mpo. The dimension of each virtual bond of mpo2 is at most Dmax.

Parameters:
- ...
- ...
Returns:
- `rho::Vector{Array{ComplexF64, 4}}`: the mpo representing the unnormalized thermal state

"""
function XTRG_algorithm()

end

end