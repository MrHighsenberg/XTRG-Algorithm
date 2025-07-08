using Test
include("../source/MPO.jl")
using .MPO
using LinearAlgebra

@testset "spin 1/2 operators" begin
    Splus, _, Id = MPO.spinlocalspace(1 // 2) 
    @test Splus == [
        0 1/sqrt(2)
        0 0
    ]
end

@testset "mpo functions for two sites" begin
    function two_site_mpo_to_matrix(mpo, d)
        return reshape(permutedims(contract(mpo[1], [1, 3], mpo[2], [3, 1]), (1,3,2,4)), (d^2, d^2))
    end
    #XY hamiltonian
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
    #identity
    d = 10
    id_mpo = identity_mpo(2, d)
    id_matrix = two_site_mpo_to_matrix(id_mpo, d)
    @test id_matrix ≈ I(d^2)
    # adding mpos
    id_mpo = identity_mpo(2, 2)
    @test two_site_mpo_to_matrix(add_mpo(H_mpo, id_mpo), 2) ≈ H_ex + I(4)
    @test two_site_mpo_to_matrix(add_mpo(id_mpo, H_mpo), 2) ≈ H_ex + I(4)
end





