module MPO
export xychain_mpo, identity_mpo, zero_mpo, mpo_to_tensor, add_mpo, square_mpo, trace_mpo, normalize_mpo!,
         leftcanonicalmpo!, rightcanonicalmpo!, sitecanonicalmpo!
using LinearAlgebra
include("../source/contractions.jl")
import .contractions: contract, tensor_svd, updateLeftEnv

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
- List of MPO tensors W[i] for sites i = 1,...,L. # leg ordering: left bottom right top

"""
function xychain_mpo(L::Int, J::Float64)
    Splus, Sminus, Id = spinlocalspace(1//2)
    Sx = (Splus + Sminus) / (sqrt(2))
    Sy = (Splus - Sminus) / (1im * sqrt(2))

    W = zeros(Complex{Float64}, 4, 2, 4, 2)
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
- List of MPO tensors W[i] for sites i = 1,...,L. # leg ordering: left bottom right top
"""
function identity_mpo(L::Int, d::Int=2)
    W = reshape(Matrix{Float64}(I, d, d), 1, d, 1, d)
    return [W for _ in 1:L]
end


"""
    zero_mpo(L::Int, d::Int)

Generates the MPO representation of the L-site zero operator.

Parameters:
- `L::Int`: Length of the spin chain.
- `d::Int`: Local Hilbert space dimension.

Returns:
- List of MPO tensors W[i] for sites i = 1,...,L. # leg ordering: left bottom right top
"""
function zero_mpo(L::Int, d::Int=2)
    W = zeros(Float64, 1, d, 1, d)
    return [W for _ in 1:L]
end


"""
    mpo_to_tensor(mpo::Vector{<:AbstractArray{<:Number, 4}})

Converts an MPO to its full tensor representation.

Parameters:
- `mpo::Vector{<:AbstractArray{<:Number, 4}}`: List of MPO tensors.

Returns:
- `T1::AbstractArray{<:Number, 4}`: Full tensor representation of the MPO.

Note: Note that the full tensor representation can only be computed efficiently for small chain lengths.
"""
function mpo_to_tensor(mpo::Vector)
    L = length(mpo)
    T1 = mpo[1]

    for itL in 2:L
        T2 = mpo[itL]

        # Local dimensions
        D1 = size(T1, 1)
        d1 = size(T1, 2)
        D2 = size(T2, 3)
        d2 = size(T2, 2)

        T1 = contract(T1, [3], T2, [1])
        T1 = permutedims(T1, (1,2,4,5,3,6))
        T1 = reshape(T1, (D1, d1*d2, D2, d1*d2))
    end
    return(T1)
end


"""
    add_mpo(A::Vector{<:AbstractArray{<:Number, 4}}, B::Vector{<:AbstractArray{<:Number, 4}})

Adds two MPOs A and B with the same physical dimensions but possibly different bond dimensions.

The function constructs a new MPO whose local tensors are block-diagonal combinations of the input MPO tensors,
increasing the bond dimension as needed. The resulting MPO represents the operator sum A + B.

Parameters:
- `A::Vector{<:AbstractArray{<:Number, 4}}`: List of MPO tensors for the first operator.
- `B::Vector{<:AbstractArray{<:Number, 4}}`: List of MPO tensors for the second operator.

Returns:
- `C::Vector{<:AbstractArray{<:Number, 4}}`: List of MPO tensors representing the sum A + B.

Note: Assumes open boundary conditions and matching physical dimensions.
"""
function add_mpo(A::Vector, B::Vector)

    L = length(A)
    T = promote_type(eltype(A[1]), eltype(B[1]))
    C = Vector{Array{T, 4}}(undef, L)

    for i in 1:L
        d1, d2 = size(A[i], 2), size(A[i], 4)
        D1l, D1r = size(A[i], 1), size(A[i], 3)
        D2l, D2r = size(B[i], 1), size(B[i], 3)
        # New tensor with summed bond dimensions
        if i == 1
            # First site: trivial left bond dimension
            C[i] = zeros(T, 1, d1, D1r + D2r, d2)
            C[i][1, :, 1:D1r, :] .= A[i][1, :, :, :]
            C[i][1, :, D1r+1:end, :] .= B[i][1, :, :, :]
        elseif i == L
            # Last site: trivial right bond dimension
            C[i] = zeros(T, D1l + D2l, d1, 1, d2)
            C[i][1:D1l, :, 1, :] .= A[i][:, :, 1, :]
            C[i][D1l+1:end, :, 1, :] .= B[i][:, :, 1, :]
        else
            # Middle sites: block diagonal structure
            C[i] = zeros(T, D1l + D2l, d1, D1r + D2r, d2)
            # Place the A block
            C[i][1:D1l, :, 1:D1r, :] .= A[i]
            # Place the B block
            C[i][D1l+1:end, :, D1r+1:end, :] .= B[i]
        end
    end
    return C
end


"""
    square_mpo(mpo::Vector{<:AbstractArray{<:Number, 4}}, Dmax::Int)

Returns mpo2 as an MPO representing the square of the operator represented by mpo. The dimension of each virtual bond of mpo2 is at most Dmax.

Parameters:
- `mpo::Vector{<:AbstractArray{<:Number, 4}}`: The MPO representing the operator to be squared, # leg ordering: left bottom right top

Returns:
- `mpo2::Vector{<:AbstractArray{<:Number, 4}}`: The MPO representing the squared operator
"""
function square_mpo(mpo::Vector)

    L = length(mpo)
    d = size(mpo[1], 2) # Assume local dimension equal at all sites

    mpo2 = Vector{Any}(undef, L)
    for i in (1:L)
        W = mpo[i]
        DL = size(W, 1)
        DR = size(W, 3)
        WW = contract(W, [4], W, [2])
        WW = permutedims(WW, (1,4,2,3,5,6))
        WW = reshape(WW, (DL^2, d, DR^2, d)) # Merging horizontal virtual legs from bottom to top
        mpo2[i] = WW
    end
    return mpo2
end


"""
    trace_mpo(mpo::Vector{<:AbstractArray{<:Number, 4}})

Computes the trace of an MPO by summing over local physical indices and contracting adjacent tensors.

Parameters:
- `mpo::Vector{<:AbstractArray{<:Number, 4}}`: The MPO representing the operator to be traced over, # leg ordering: left bottom right top

Returns:
- `trace::Float64`: The value of the trace.
"""
function trace_mpo(mpo::Vector)
    L = length(mpo)
    
    matrices = Vector{Matrix{Any}}(undef, L)
    
    for i in 1:L
        matrices[i] = zeros(size(mpo[i], 1), size(mpo[i], 3))
        
        # Tracing over physical indices
        @assert size(mpo[i], 2) == size(mpo[i], 4)
        for j in 1:size(mpo[i], 2)
            matrices[i] .+= mpo[i][:, j, :, j]
        end
    end
    
    # Contract matrices from left to right
    result = matrices[1]
    for i in 2:L
        result = result * matrices[i]
    end
    
    # Extract scalar trace value
    return result[1,1]
end


"""
    normalize_mpo!(mpo::Vector{<:AbstractArray{<:Number, 4}})

Normalizes an MPO in-place by distributing the normalization weight across all tensors.
Uses the updateLefEnv function to iteratively contract the MPO tensors to compute the Frobenius norm.

Parameters:
- `mpo::Vector{<:AbstractArray{<:Number, 4}}`: Vector of local MPO tensors 

Returns:
- `norm::Float64`: Norm of the unnormalized input MPO

"""
function normalize_mpo!(mpo::Vector)
    L = length(mpo)
    
    # Initialize left environment as identity
    V = ones(ComplexF64, 1, 1, 1)
    
    # Contract from left to right tracing out physical indices at each step
    for i in 1:L
        
        A = mpo[i]
        C = conj(permutedims(mpo[i], (1,4,3,2)))
        
        # Create trivial middle tensor for updateLeftEnv function
        d = size(A, 2)
        B = zeros(ComplexF64, 1, d, 1, d)
        for j in 1:d
            B[1, j, 1, j] = 1.0
        end
        
        # Update left environment
        V = updateLeftEnv(V, A, B, C)
    end
    
    # Evaluate the Frobenius norm
    norm_squared = abs(V[1, 1, 1])
    norm = sqrt(norm_squared)
    
    # Normalize the MPO by distributing the weight    
    for i in 1:L
        mpo[i] .*= norm^(-1/L)
    end 
    return norm
end


"""
    leftcanonicalmpo!(mpo::Vector{<:AbstractArray{<:Number, 4}}; Nkeep::Int=typemax(Int), tolerance::Float64=0.0)

Modifies the MPO in-place to bring it into left-canonical form.

Parameters:
- `mpo::Vector{<:AbstractArray{<:Number, 4}}`: Vector of local MPO tensors
- `Nkeep::Int`: maximum number of singular values to keep. Default is `typemax(Int)`.
- `tolerance::Float64`: minimum magnitude of singular values to keep. Default is `0.0`.
"""
function leftcanonicalmpo!(mpo::Vector; Nkeep::Int=typemax(Int), tolerance::Float64=0.0)
    L = length(mpo)
    # Left-canonicalize site by site from left to right
    for itL in 1:L-1
        U, S, Vd, _ = tensor_svd(mpo[itL], [1,2,4]; Nkeep = Nkeep, tolerance = tolerance)
        mpo[itL] = permutedims(U, (1,2,4,3))
        mpo[itL+1] = contract(Diagonal(S)*Vd, [2], mpo[itL+1], [1])
    end
end


"""
    rightcanonicalmpo!(mpo::Vector{<:AbstractArray{<:Number, 4}}; Nkeep::Int=typemax(Int), tolerance::Float64=0.0)

Modifies the MPO in-place to bring it into right-canonical form.

Parameters:
- `mpo::Vector{<:AbstractArray{<:Number, 4}}`: Vector of local MPO tensors
- `Nkeep::Int`: maximum number of singular values to keep. Default is `typemax(Int)`.
- `tolerance::Float64`: minimum magnitude of singular values to keep. Default is `0.0`.
"""
function rightcanonicalmpo!(mpo::Vector; Nkeep::Int=typemax(Int), tolerance::Float64=0.0)
    L = length(mpo)
    # Right-canonicalize site by site from right to left
    for itL in L:-1:2
        U, S, Vd, _ = tensor_svd(mpo[itL], [1]; Nkeep = Nkeep, tolerance = tolerance)
        mpo[itL] = Vd
        mpo[itL-1] = permutedims(contract(mpo[itL-1], [3], U*Diagonal(S), [1]), (1,2,4,3))
    end
end


"""
    sitecanonicalmpo!(mpo::Vector{<:AbstractArray{<:Number, 4}}, l::Int; Nkeep::Int=typemax(Int), tolerance::Float64=0.0)

Modifies the MPO in-place to bring it into site-canonical form with respect to site l, where tensors to the left of site l
are in left-canonical form and tensors to the right of site l are in right-canonical form.

Parameters:
- `mpo::Vector{<:AbstractArray{<:Number, 4}}`: Vector of local MPO tensors
- `l::Int`: Index of the orthogonality center
- `Nkeep::Int`: maximum number of singular values to keep. Default is `typemax(Int)`.
- `tolerance::Float64`: minimum magnitude of singular values to keep. Default is `0.0`.
"""
function sitecanonicalmpo!(mpo::Vector, l::Int; Nkeep::Int=typemax(Int), tolerance::Float64=0.0)
    L = length(mpo)

    # Check validty of orthogonality center index
    if l < 1 || l > L
        error("Site index l must be between 1 and L=$L")
    end
    
    # First apply left canonicalization to get everything in left-canonical form
    leftcanonicalmpo!(mpo; Nkeep = Nkeep, tolerance = tolerance)
    
    # Right-canonicalize tensors from site L to l+1
    for itL in L:-1:l+1
        U, S, Vd, _ = tensor_svd(mpo[itL], [1]; Nkeep = Nkeep, tolerance = tolerance)
        mpo[itL] = Vd
        mpo[itL-1] = permutedims(contract(mpo[itL-1], [3], U*Diagonal(S), [1]), (1,2,4,3))
    end
end

end