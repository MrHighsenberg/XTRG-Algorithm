module MPO
export xychain_mpo, identity_mpo, add_mpo, square_mpo, leftcanonicalmpo
using LinearAlgebra
include("../source/contractions.jl")
import .contractions: contract, tensor_svd


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
    W = reshape(Matrix{Float64}(I, d, d), 1, d, 1, d) # Ordering: left bottom right top
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
    square_mpo(mpo::Vector, Dmax::Int)

Returns mpo2 as an MPO representing the square of the operator represented by mpo. The dimension of each virtual bond of mpo2 is at most Dmax.

Parameters:
- `mpo::Vector{Array{ComplexF64, 4}}`: the mpo representing the operator to be squared, # leg ordering for each tensor: left bottom right top
- `Dmax::Int`: maximal bond dimension of output mpo
Returns:
- `mpo2::Vector{Array{ComplexF64, 4}}`: the mpo representing the squared operator if maximal bond dimension is not exceeded (else return mpo)

"""
function square_mpo(mpo::Vector, Dmax::Int=100)
    L = length(mpo)
    d = size(mpo[1], 2) # assume local dimension equal at all sites
    D = max(maximum([size(W, 1) for W in mpo]), maximum([size(W, 3) for W in mpo])) # maximal bond dimension across all W tensors
    if D^2 <= Dmax
        mpo2 = Vector{Any}(undef, L)
        for i in (1:L)
            W = mpo[i]
            DL = size(W, 1)
            DR = size(W, 3)
            WW = contract(W, [4], W, [2])
            WW = permutedims(WW, (1,4,2,3,5,6))
            WW = reshape(WW, (DL^2, d, DR^2, d)) # merging horizontal virtual legs from bottom to top
            mpo2[i] = WW
        end
        return mpo2
    else # square mpo requires too much storage
        error("Maximal bond dimension D = $Dmax reached")
    end
end


"""
    leftcanonicalmpo(mpo::Vector; Nkeep::Int=typemax(Int), tolerance::Float64=0.0)

Returns mpo in left-canonical form.

Parameters:
- `mpo::AbstractVector{<:AbstractArray{<:Number, 4}}`: List of MPO tensors
- `Nkeep::Int`: maximum number of singular values to keep. Default is `typemax(Int)`.
- `tolerance::Float64`: minimum magnitude of singular values to keep. Default is `0.0`.

Returns:
- `leftmpo::Vector`: mpo in left-canonical form.
"""

function leftcanonicalmpo(mpo::Vector; Nkeep::Int=typemax(Int), tolerance::Float64=0.0)
    L = length(mpo)
    leftmpo = deepcopy(mpo)
    # left canonicalize site by site
    for itL in 1:L-1
        U, S, Vd, _ = tensor_svd(leftmpo[itL], [1,2,4]; Nkeep=Nkeep, tolerance=tolerance)
        leftmpo[itL] = permutedims(U, (1,2,4,3))
        leftmpo[itL+1] = contract(Diagonal(S)*Vd, [2], leftmpo[itL+1], [1])
    end
    return(leftmpo)
end

# #### LATER DISCARD FROM HERE

# """
# Copying below here svd and svdleft for MPS tensors, copying from lecture

# """

# function svd(
#     T::AbstractArray{<:Number}, indicesU::AbstractVector{Int};
#     Nkeep::Int=typemax(Int), tolerance::Float64=0.0
# )
#     if isempty(T)
#         return (zeros(0, 0), zeros(0), zeros(0, 0), 0.0)
#     end

#     indicesV = setdiff(1:ndims(T), indicesU)
#     Tmatrix = reshape(
#         permutedims(T, cat(dims=1, indicesU, indicesV)),
#         (prod(size(T)[indicesU]), prod(size(T)[indicesV]))
#     )
#     svdT = LinearAlgebra.svd(Tmatrix)

#     Nkeep = min(Nkeep, size(svdT.S, 1))
#     Ntolerance = findfirst(svdT.S .< tolerance)
#     if !isnothing(Ntolerance)
#         Nkeep = min(Nkeep, Ntolerance - 1)
#     end

#     U = reshape(svdT.U[:, 1:Nkeep], size(T)[indicesU]..., Nkeep)
#     S = svdT.S[1:Nkeep]
#     Vd = reshape(svdT.Vt[1:Nkeep, :], Nkeep, size(T)[indicesV]...)
#     discardedweight = sum(svdT.S[Nkeep+1:end] .^ 2)

#     return (U, S, Vd, discardedweight)
# end
# function svdleft(
#     T::AbstractArray{<:Number};
#     Nkeep::Int=typemax(Int), tolerance::Float64=0.0
# )
#     U, S, Vd, _ = svd(T, collect(1:ndims(T)-1), Nkeep=Nkeep, tolerance=tolerance)
#     return U, Diagonal(S) * Vd
# end


# """
#     leftcanonicalmpo_old(mpo::Vector; Nkeep::Int=typemax(Int), tolerance::Float64=0.0)

# Returns mpo in left-canonical form.

# Parameters:
# - `mpo::AbstractVector{<:AbstractArray{<:Number, 4}}`: List of MPO tensor
# - `Nkeep::Int`: maximum number of singular values to keep. Default is `typemax(Int)`.
# - `tolerance::Float64`: minimum magnitude of singular value to keep. Default is `0.0`.

# Returns:
# - `leftmpo::Vector`: mpo in left-canonical form.
# """
# function leftcanonicalmpo_old(mpo::AbstractVector{<:AbstractArray{<:Number, 4}}; Nkeep::Int=typemax(Int), tolerance::Float64=0.0)
#     L = length(mpo)
#     mps = []
#     #transform mpo to an mps by fusing physical legs
#     for itL in 1:L
#         W = permutedims(mpo[itL], (1,2,4,3))
#         Dleft = size(W,1)
#         Dright = size(W,4)
#         d = size(W,2)
#         push!(mps, reshape(W, (Dleft, d^2, Dright)))
#     end
#     #left canonicalize the mps in place (copying from lecture. i later realized we could simply use tensor_svd and avoid using mps at all.)
#     for itL in 1:L-1
#         A, Lambda = svdleft(mps[itL]; Nkeep=Nkeep, tolerance=tolerance)
#         mps[itL] = A
#         mps[itL+1] = contract(Lambda, [2], mps[itL+1], [1])
#     end
#     #transform left canonicalized mps to left canonicalized mpo
#     leftmpo = []
#     for itL in 1:L
#         M = mps[itL]
#         Dleft = size(M,1)
#         Dright = size(M,3)
#         d = Int(sqrt(size(M, 2)))
#         push!(leftmpo, permutedims(reshape(M, (Dleft, d, d, Dright)), (1,2,4,3)))
#     end
#     return(leftmpo)
# end

# #### LATER DISCARD UP TO HERE

end