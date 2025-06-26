module MPO
export xychain_mpo, identity_mpo, add_mpo
using LinearAlgebra

"""
    spinlocalspace(spin::Rational=1//2)

Constructs the spin local space operators for a given spin value.

Parameters:
- `spin::Rational`: The spin value, which should be a positive (half-)integer.
Returns:
- `Splus`: The spin raising operator.
- `Sminus`: The spin lowering operator.

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
- List object of MPO tensors W[i] on sites i = 1,...,L.

"""
function xychain_mpo(L::Int, J::Float64)
    Splus, Sminus, Id = spinlocalspace(1 // 2)
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
- List of MPO tensors representing the sum A + B.

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

end