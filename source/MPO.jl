module MPO
export xychain_mpo, identity_mpo, add_mpo, square_mpo
using LinearAlgebra
include("../source/contractions.jl")
import .contractions: contract


"""
    spinlocalspace(spin::Rational=1//2)

Constructs the spin local space operators for a given spin value.

Parameters:
- `spin::Rational`: The spin value, which should be a positive (half-)integer.
Returns:
- `Splus`: The spin raising operator.
- `Sminus`: The spin lowering operator.
- `Id`: The identity operator.

The spin local space operators are represented as matrices of size `(2spin + 1, 2spin + 1)`.
"""
function spinlocalspace(spin::Rational=1//2)
    if !isinteger(2 * spin)
        error("Spin should be positive (half-)integer.")
    end

    spinvals = collect(spin:-1:-spin)
    Splus = diagm(1 =>
        sqrt.((spin .- spinvals[2:end]) .* (spin .+ spinvals[1:end-1]))
    ) / sqrt(2)

    Sminus = diagm(-1 =>
        sqrt.((spin .+ spinvals[1:end-1]) .* (spin .- spinvals[2:end]))
    ) / sqrt(2)

    Id = I(size(Splus, 1))

    return Splus, Sminus, Id
end

"""
    xychain_mpo(L::Int, J::Float64)

Generates the MPO representation of the XY-chain.

Parameters:
- `L::Int`: Length of the spin chain.
- `J::Float64`: Coupling constant of the spin-spin interaction.
Returns:
- List of MPO tensors W[i] for sites i = 1,...,L.

"""
function xychain_mpo(L::Int, J::Float64)
    Splus, Sminus, Id = spinlocalspace(1//2)
    Sx = (Splus + Sminus) / (sqrt(2))
    Sy = (Splus - Sminus) / (1im * sqrt(2))

    W = zeros(Complex{Float64}, 4, 2, 4, 2) # Ordering: left bottom right top
    W[1, :, 1, :] = Id
    W[2, :, 1, :] = Sx
    W[3, :, 1, :] = Sy
    W[4, :, 2, :] = J*Sx
    W[4, :, 3, :] = J*Sy
    W[4, :, 4, :] = Id

    return [
        W[[4], :, :, :], # First site
        [W for _ in 2:L-1]...,
        W[:, :, [1], :] # Last site
    ]
end

"""
    identity_mpo(L::Int, d::Int)

Generates the MPO representation of the L-site identity operator.

Parameters:
- `L::Int`: Length of the spin chain.
- `d::Int`: Local Hilbert space dimension.

Returns:
- List of MPO tensors W[i] for sites i = 1,...,L.
"""
function identity_mpo(L::Int, d::Int=2)
    W = reshape(Matrix{ComplexF64}(I, d, d), 1, d, 1, d) # Ordering: left bottom right top
    return [W for _ in 1:L]
end

"""
    add_mpo(A::Vector, B::Vector)

Adds two MPOs A and B with the same physical dimensions but possibly different bond dimensions.

The function constructs a new MPO whose local tensors are block-diagonal combinations of the input MPO tensors,
increasing the bond dimension as needed. The resulting MPO represents the operator sum A + B.

Parameters:
- `A::Vector`: List of MPO tensors for the first operator.
- `B::Vector`: List of MPO tensors for the second operator.

Returns:
- `C::Vector`: List of MPO tensors representing the sum A + B.

Note: Assumes open boundary conditions and matching physical dimensions.
"""
function add_mpo(A::Vector, B::Vector)
    L = length(A)
    C = Vector{Any}(undef, L)
    for i in 1:L
        d1, d2 = size(A[i], 2), size(A[i], 4)
        D1l, D1r = size(A[i], 1), size(A[i], 3)
        D2l, D2r = size(B[i], 1), size(B[i], 3)
        # New tensor with summed bond dimensions
        C[i] = zeros(eltype(A[i]), D1l + D2l, d1, D1r + D2r, d2)
        # Place the A block
        C[i][1:D1l, :, 1:D1r, :] .= A[i]
        # Place the B block
        C[i][D1l+1:end, :, D1r+1:end, :] .= B[i]
    end
    return C
end

"""
    square_mpo(mpo, Dmax, Nsweep)

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

end