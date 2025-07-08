using Test
include("../source/contractions.jl")
import .contractions: contract

@testset "contract function" begin
    @testset "Dimensions of contracted tensors" begin
        A = ones(3, 4, 5)
        B = ones(1, 3, 4, 10)

        C1 = contract(A, [1], B, [2])
        @test size(C1) == (4, 5, 1, 4, 10)
        @test all(C1 .== 3)

        C2 = contract(A, [1, 2], B, [2, 3])
        @test size(C2) == (5, 1, 10)
        @test all(C2 .== 12)
    end

    @testset "Contracting matrices is the same as matrix multiplication" begin
        A = [0.917561 0.198191; 0.892127 0.616371]
        B = [0.0598884 0.722894; 0.927395 0.533697]
        Id = [1 0; 0 1]
        @test contract(A, 2, B, 1) == A * B
        @test contract(B, 1, A, 2, (2, 1)) == A * B
        @test contract(A, 2, Id, 1) == A
    end
end
