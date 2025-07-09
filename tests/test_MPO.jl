using Test
using LinearAlgebra
include("../source/MPO.jl")
import .contractions: contract
import .MPO: spinlocalspace, xychain_mpo, identity_mpo, add_mpo, square_mpo

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
        @test two_site_mpo_to_matrix(add_mpo(id_mpo, H_mpo), 2) ≈ H_ex + I(4)
    end
end

@testset "square_mpo" begin
    L = 3
    D = 10
    d = 10
    mpo = vcat([rand(ComplexF64, 1, d, D, d)], [rand(ComplexF64, D, d, D, d) for _ in (2:L-1)], [rand(ComplexF64, D, d, 1, d)])
    mpo2 = square_mpo(mpo, D^2)

    function get_matrix(M, d)
        M1, M2, M3 = M
        T = contract(M1, [3], M2, [1])
        T = contract(T, [5], M3, [1])
        T = permutedims(T, (1,2,4,6,7,3,5,8))
        return reshape(T, (d^3, d^3))
    end

    mat1 = get_matrix(mpo, d)^2
    mat2 = get_matrix(mpo2, d)
    @test mat1 ≈ mat2
end