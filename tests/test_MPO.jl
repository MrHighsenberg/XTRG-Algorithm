using Test
using LinearAlgebra
include("../source/MPO.jl")
import .contractions: contract
import .MPO: spinlocalspace, xychain_mpo, identity_mpo, mpo_to_tensor, add_mpo, square_mpo, normalize_mpo!, leftcanonicalmpo

@testset "spinlocalspace" begin
    Splus, _, Id = spinlocalspace(1//2) 
    @test Splus == [
        0 1/sqrt(2)
        0 0
    ]
    @test Id == [
        1 0
        0 1
    ]
end

# Testing MPO operations on two sites
function two_site_mpo_to_matrix(mpo, d)
    return reshape(permutedims(contract(mpo[1], [1, 3], mpo[2], [3, 1]), (1,3,2,4)), (d^2, d^2))
end
    
@testset "xychain_mpo" begin
    J = 0.38
    H_mpo = xychain_mpo(2, J)
    H = two_site_mpo_to_matrix(H_mpo, 2)
    H_ex = [
        0 0 0 0
        0 0 J/2. 0
        0 J/2. 0 0
        0 0 0 0
    ]
    @test H ≈ H_ex
end

@testset "identity_mpo" begin
    d = 10
    id_mpo = identity_mpo(2, d)
    id_matrix = two_site_mpo_to_matrix(id_mpo, d)
    @test id_matrix ≈ I(d^2)
end

@testset "add_mpo" begin
    J = 0.63
    H_mpo = xychain_mpo(2, J)
    H_ex = [
        0 0 0 0
        0 0 J/2. 0
        0 J/2. 0 0
        0 0 0 0
    ]
    id_mpo = identity_mpo(2, 2)
    @test two_site_mpo_to_matrix(add_mpo(H_mpo, id_mpo), 2) ≈ H_ex + I(4)
end

@testset "mpo_to_tensor" begin
    @testset "Identity MPO" begin
        id_mpo = identity_mpo(2, 2)
        full_tensor = mpo_to_tensor(id_mpo)
        @test size(full_tensor) == (1, 4, 1, 4)
        @test reshape(full_tensor, 4, 4) ≈ I(4)
    end
    
    @testset "Two-site tensor product" begin
        sigma_x = [0.0 1.0; 1.0 0.0]
        mpo = [reshape(sigma_x, 1, 2, 1, 2), reshape(sigma_x, 1, 2, 1, 2)]
        full_tensor = mpo_to_tensor(mpo)
        @test reshape(full_tensor, 4, 4) ≈ kron(sigma_x, sigma_x)
    end
end

@testset "square_mpo" begin
    L = 3
    D = 10
    d = 10
    mpo = vcat([rand(ComplexF64, 1, d, D, d)], [rand(ComplexF64, D, d, D, d) for _ in (2:L-1)], [rand(ComplexF64, D, d, 1, d)])
    mpo2 = square_mpo(mpo, D^2)
    mat1 = reshape(mpo_to_tensor(mpo), (d^3, d^3))^2
    mat2 = reshape(mpo_to_tensor(mpo2), (d^3, d^3))
    @test mat1 ≈ mat2
end

@testset "normalize_mpo!" begin
    L = 4
    # Generate a random MPO
    mpo = [rand(ComplexF64, 1, 5, 2, 5), rand(ComplexF64, 2, 8, 5, 8), rand(ComplexF64, 5, 3, 7, 3), rand(ComplexF64, 7, 4, 1, 4)]
    
    # Store original tensor for comparison
    original_tensor = mpo_to_tensor(mpo)
    original_norm = norm(reshape(original_tensor, :))
    
    mpo_copy = deepcopy(mpo)
    
    # Normalize the MPO
    normalize_mpo!(mpo)
    
    # Check that the normalized MPO has unit Frobenius norm
    normalized_tensor = mpo_to_tensor(mpo)
    normalized_norm = norm(reshape(normalized_tensor, :))
    @test normalized_norm ≈ 1.0
    
    # Check that the normalized MPO represents the same operator up to scaling
    @test normalized_tensor ≈ original_tensor / original_norm
    
    # Check that the original MPO tensors were modified in-place
    @test !(mpo[1] ≈ mpo_copy[1])
    @test !(mpo[3] ≈ mpo_copy[3])
end

@testset "leftcanonicalmpo" begin
    L = 4
    # Generate a random MPO
    mpo = [rand(ComplexF64, 1, 5, 2, 5), rand(ComplexF64, 2, 2, 4, 2), rand(ComplexF64, 4, 3, 7, 3), rand(ComplexF64, 7, 2, 1, 2)]
    
    # Normalize the MPO 
    normalize_mpo!(mpo)

    # Left-canonicalize the MPO
    leftmpo = leftcanonicalmpo(mpo)
    for itL in 1:L
        W = leftmpo[itL]
        @test contract(conj(W), [1,2,4], W, [1,2,4]) ≈ I(size(W, 3)) # Checking the isometry property
    end
    @test mpo_to_tensor(mpo) ≈ mpo_to_tensor(leftmpo) # Checking that both MPOs represent the same operator
    @test !(mpo[3][1,1,1,1] == leftmpo[3][1,1,1,1]) # but local tensors are different
end