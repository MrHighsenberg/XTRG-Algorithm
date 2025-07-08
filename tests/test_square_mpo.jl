using Test
include("../source/MPO.jl")
include("../source/contractions.jl")
import .MPO: square_mpo
import .contractions: contract
using LinearAlgebra

@testset "mpo squaring" begin
    L = 3
    D = 10
    d = 10
    mpo = vcat([rand(ComplexF64, 1, d, D, d)], [rand(ComplexF64, D, d, D, d) for _ in (2:L-1)], [rand(ComplexF64, D, d, 1, d)])
    Nsweep = 10
    mpo2_exact = square_mpo(mpo, D^2, Nsweep)
    # mpo2_truncated = square_mpo(mpo, 2 * D - 1, Nsweep)
    function get_matrix(M, d)
        M1, M2, M3 = M
        T1 = contract(M1, [3], M2, [1])
        T2 = contract(T1, [5], M3, [1])
        T3 = permutedims(T2, (1,2,4,3,5,6,7,8))
        T4 = permutedims(T3, (1,2,3,6,5,4,7,8))
        T5 = permutedims(T4, (1,2,3,4,6,5,7,8))
        return reshape(T5, (d^3, d^3))
    end
    mat1 = get_matrix(mpo, d)^2
    mat2 = get_matrix(mpo2_exact, d)
    # mat3 = get_matrix(mpo2_truncated, d)
    @test mat1 ≈ mat2
    # @test norm(mat1 - mat3) < 1e-3 # !! not the best test since you may get ulucky with the random mpo. make sure this test is never failed even with random numbers, otherwise consider changing from random mpo to a fixed one.
end



