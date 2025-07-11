using Test
using LinearAlgebra
include("../source/contractions.jl")
import .contractions: contract, tensor_svd, updateLeft, updateLeftEnv

@testset "contract" begin
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

@testset "tensor_svd" begin
    @testset "Obtaining isometries using tensor_svd" begin
        d = 3
        D = 30
        M = rand(D, d, D)
        # SVD with left leg as U
        U, S, Vd, discardedweight = tensor_svd(M, [1])
        @test size(U) == (D, D)
        @test size(S) == (D,)
        @test size(Vd) == (D, d, D)
        @test contract(contract(U, [2], Diagonal(S), [1]), [2], Vd, [1]) ≈ M
        # Check left isometry condition: U'U = I
        @test contract(U, [1], conj(U), [1]) ≈ I(D)
        # Check right isometry condition: Vd Vd' = I
        @test contract(Vd, [2, 3], conj(Vd), [2, 3]) ≈ I(D)
    end

    @testset "tensor_svd with tolerance" begin
        T = diagm([3.0, 2.0, 1.0, 0.1])
        T = reshape(T, (2, 2, 2, 2))
        U, S, Vd, discardedweight = tensor_svd(T, [1, 2], tolerance=1.5)
        @test length(S) == 2
        @test isapprox(discardedweight, 1.0^2 + 0.1^2; atol=1e-12)
    end

    @testset "tensor_svd on empty tensor" begin
        T = zeros(0, 0)
        U, S, Vd, discardedweight = tensor_svd(T, [1])
        @test size(U) == (0, 0)
        @test size(S) == (0,)
        @test size(Vd) == (0, 0)
        @test discardedweight == 0.0
    end

    @testset "tensor_svd on many-leg tensor" begin
        M = rand(ComplexF64, 3, 5, 2, 5, 7)
        U, S, Vd, _ = tensor_svd(M, [1, 2, 4])
        @test M ≈ contract(U, [4], contract(Diagonal(S), [2], Vd, [1]), [1], (1,2,4,3,5))
    end
end

@testset "updateLeft" begin
    @testset "updateLeft dimensions" begin
        D = 4
        w = 2
        d = 3

        A = ones(D, d, D)
        C = ones(D, w, D)
        X = ones(w, d, w, d)

        C_new = updateLeft(C, A, X, A)
        @test size(C_new) == (D, w, D)
        @test all(C_new .== D^2 * w * d^2)
    end

    @testset "updateLeft applied to isometry" begin
        # This is a randomly generated isometry.
        U = [
            -0.2664622767440691 - 0.1663280646719271im   -0.07471385066622838 + 0.4866024066525218im     0.12343643370564508 - 0.2930965513192281im    -0.4771806928942692 - 0.06822925300977367im
            -0.25832662599939993 - 0.1586291018793261im    -0.2963047100642419 - 0.10213982581092121im    -0.3344032594502972 + 0.25559207738522327im  -0.19879122457090895 + 0.11212346256144301im
            -0.30461959897756735 - 0.2632149079545921im    -0.2921982724835089 - 0.13794571235099565im  -0.027027702687684972 - 0.5358418545160035im   -0.04096547761356491 + 0.23525532573444521im
            -0.29542902164702417 - 0.21341371044601504im    0.1721500002173616 + 0.19476471220148406im    -0.0724127053677811 + 0.29562878664271436im    -0.072021894991373 - 0.09001874477106447im
            -0.31910591470575317 - 0.16653786590153427im   0.24527423240540716 - 0.4203307889186253im    0.006722007012845857 - 0.07073057848514176im  -0.04584517603637932 - 0.16681288565541413im
            -0.29164510980422903 - 0.26625718847361135im    0.3190371268356846 + 0.17908600844430286im    0.16380121473436773 - 0.15099006211532753im   0.38144437122780595 + 0.1148541894792461im
            -0.2438824933377057 - 0.2827783920579037im   0.012764017127802106 + 0.04351021859772729im   -0.15803418676807654 + 0.37087062025650497im   0.06587109159032137 - 0.39432096706018577im
            -0.23381205216253428 - 0.18685425991767043im  -0.22355083545381035 - 0.2543702109930897im      0.2551309804804418 + 0.24273352478903604im   0.34119745419605146 + 0.4247076952910319im
        ]
        A = reshape(U, (4, 2, 4))
        C = reshape(I(4), (4, 1, 4))
        X = reshape(I(2), (1, 2, 1, 2))
        @test updateLeft(C, A, X, A) ≈ reshape(I(4), (4, 1, 4))
    end
end

@testset "updateLeftEnv" begin
    @testset "updateLeftEnv dimensions" begin
        Da = 3
        Db = 3
        Dc = 9
        d = 2

        V = ones(Dc, Db, Da)
        A = ones(Da, d, Da, d)
        B = ones(Db, d, Db, d)
        C = ones(Dc, d, Dc, d)

        V_new = updateLeftEnv(V, A, B, C)
        @test size(V_new) == (Dc, Db, Da)
        @test all(V_new .== Da * Db * Dc * d^3)
    end

    @testset "updateLeftEnv applied to isometry" begin
        # This is another randomly generated isometry.
        U = [
            -0.15749891104736813 - 0.30508368177584466im	-0.15324927380298695 + 0.2253958711510132im	-0.25468447240514075 - 0.07963018090540669im	-0.2448374080800449 + 0.07880973555796383im
            -0.1650117127599791 + 0.1936879424107124im	-0.3314315705793687 + 0.02572851680625796im	0.11909134280819163 + 0.1307050281733177im	-0.3318385223283171 + 0.001833364125427478im
            0.5199588209315036 + 0.4707902259955522im	0.017704431765831183 + 0.19734011030087123im	-0.4828023886586157 - 0.09690016306139951im	0.07177695892046267 + 0.07426934010109275im
            -0.02983410619995426 + 0.30419654031556137im	0.3403544516328344 - 0.36432837944629193im	0.037737248672448154 + 0.25386652909755975im	-0.5702597122849564 + 0.24291938528767326im
            -0.13343718864192225 - 0.14257072776758806im	0.17605652735156302 - 0.10703177015478504im	-0.5296537643978843 - 0.15059280399118988im	0.06733844957830346 + 0.3132982238181955im
            0.09694921211136276 - 0.08937438071710804im	-0.23351174864659396 + 0.20831337223151464im	-0.15447418850412883 + 0.41371914390725195im	-0.3192897646023946 + 0.22966616396169964im
            0.03929596547909546 - 0.2042001391234323im	-0.0522697291836111 - 0.5917601204300642im	-0.10897006526476426 - 0.18289770736039745im	-0.07607054344272768 + 0.07666740696504934im
            0.3059014219131799 + 0.19902652919571348im	-0.17415053960130927 - 0.061279346630354115im	0.16170557527699536 + 0.13948602651357145im	-0.10563269267344981 - 0.3855438237428369im
        ]
        V = reshape(I(4), (4, 1, 4))
        A = reshape(U, (4, 2, 4, 1))
        B = reshape(I(2), (1, 2, 1, 2))
        C = permutedims(reshape(conj(U), (4, 2, 4, 1)), (1, 4, 3, 2))
        @test updateLeftEnv(V, A, B, C) ≈ reshape(I(4), (4, 1, 4))
    end
end