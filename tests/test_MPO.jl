using Test
using LinearAlgebra
include("../source/MPO.jl")
import .contractions: contract
import .MPO: spinlocalspace, xychain_mpo, identity_mpo, add_mpo, square_mpo, leftcanonicalmpo

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

@testset "mpo operations" begin
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
end

function mpo_to_tensor(mpo)
        L = length(mpo)
        T1 = mpo[1]
        D1 = size(T1, 1)
        d1 = size(T1, 2)
        for itL in 2:L
            T2 = mpo[itL]
            D2 = size(T2, 3)
            d2 = size(T2, 2)
            T1 = contract(T1, [3], T2, [1])
            T1 = permutedims(T1, (1,2,4,3,5,6))
            T1 = permutedims(T1, (1,2,3,5,4,6))
            T1 = reshape(T1, (D1, d1*d2, D2, d1*d2))
            d1 = d1 * d2
        end
        return(T1)
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

@testset "leftcanonicalmpo" begin

    L = 4
    # Generate random mpo and left-canonicalize it
    mpo = [rand(ComplexF64, 3, 5, 2, 5), rand(ComplexF64, 2, 2, 4, 2), rand(ComplexF64, 4, 3, 7, 3), rand(ComplexF64, 7, 2, 2, 2)]

    leftmpo = leftcanonicalmpo(mpo)
    for itL in 1:L # Only test for sites 1 to L-1 (left-canonical sites)
        W = leftmpo[itL]
        @test contract(conj(W), [1,2,4], W, [1,2,4]) ≈ I(size(W, 3)) # Checking the isometry property
    end
    @test mpo_to_tensor(mpo) ≈ mpo_to_tensor(leftmpo) # Checking that mpo and leftmpo represent the same operator
    @test !(mpo[3][1,1,1,1] == leftmpo[3][1,1,1,1]) # but local tensors are different
end