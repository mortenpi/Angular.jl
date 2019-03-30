using Angular
using Angular: halfint
using LinearAlgebra
using WignerSymbols
using WignerSymbols: HalfInteger
using AtomicLevels: terms, _terms_jw, Term, Orbital, weight
using Test

# ==========================================================================================
# TestBasis
# ------------------------------------------------------------------------------------------
countmatrix(i::Integer, j::Integer) = div((max(i, j) - 1)*max(i, j), 2) + min(i, j)
countmatrix(i::Integer) = countmatrix(i, i)

using Angular: BasisSet, LinearOperator
struct TestBasis <: BasisSet
    n :: Int

    function TestBasis(n)
        @assert n > 0
        new(n)
    end
end
Base.length(b::TestBasis) = b.n

struct TestOperator <: LinearOperator{TestBasis}
    b :: TestBasis
end
Angular.basis(op::TestOperator) = op.b
function Angular.apply(i::Integer, op::TestOperator, j::Integer) :: ComplexF64
    (1 <= i <= length(op.b)) && (1 <= j <= length(op.b)) || throw(BoundsError(op, (i, j)))
    countmatrix(i, j)
end
# ==========================================================================================

@testset "Angular.jl" begin
    @testset "TestBasis" begin
        b = TestBasis(3)
        @test length(b) == 3
        op = TestOperator(b)

        @test Angular.matrix(op) == [1 2 4; 2 3 5; 4 5 6]

        v1 = Angular.apply(op, [1, 1, 1])
        v2 = Angular.apply(op, v1)
        v3 = Angular.apply(op, v1 + v2)
        @test v1.cs == [7, 10, 15]
        @test v2.cs == [87, 119, 168]
        @test v3.cs == [1084, 1490, 2119]

        # identity operator
        id = Angular.IdentityOperator(b)
        @test Angular.matrix(id) == I

        # TODO: add tests for compositeoperators
    end
    @testset "angular momentum" begin
        mz = Angular.matrix(Angular.JOperator(Angular.AngularBasis(5//2), :z))
        @test isdiag(mz)
        for twoj = 0:20
            b = Angular.AngularBasis(twoj//2)
            @test length(b) == twoj+1
            j = convert(Float64, b.j)
            Jz, J₊, J₋ = Angular.JOperator(b, :z), Angular.JOperator(b, :+), Angular.JOperator(b, :-)
            Jx = (J₊ + J₋) / 2
            Jy = (J₊ - J₋) / (2im)
            J2matrix = Angular.matrix(Jx^2 + Jy^2 + Jz^2)
            @test isdiag(J2matrix)
            @test all(diag(J2matrix) .≈ j*(j + 1))

            # Test the simplified version of the J^2 operator
            @test J2matrix ≈ Angular.matrix((J₊*J₋ + J₋*J₊)/2 + Jz^2)
        end

        for twoj = 0:20
            j = twoj / 2
            λ = j * (j + 1)
            @test Angular.findj(λ) == twoj // 2
        end
    end
    @testset "JOperatorSet" begin
        for twoj = 0:10
            J = Angular.JOperatorSet(Angular.AngularBasis(twoj//2))
            Jz, J₊, J₋ = J.Jz, J.J₊, J.J₋
            @test Angular.matrix(Angular.Jx(J)) ≈ Angular.matrix((J₊ + J₋) / 2)
            @test Angular.matrix(Angular.Jy(J)) ≈ Angular.matrix((J₊ - J₋) / (2im))
            @test Angular.matrix(Angular.J2(J)) ≈ Angular.matrix((J₊*J₋ + J₋*J₊)/2 + Jz^2)
        end
    end

    @testset "tensor products" begin
        b1 = TestBasis(4)
        b2 = Angular.AngularBasis(5//2)
        op1 = TestOperator(b1)
        op2 = Angular.JOperator(b2, :z)

        # tensor products
        tpb = Angular.TensorProductBasis(b1, b2)
        @test length(tpb) == length(b1) * length(b2)

        @test Angular.leftindex(tpb, 1) == 1
        @test Angular.rightindex(tpb, 1) == 1
        @test Angular.leftindex(tpb, 7) == 3
        @test Angular.rightindex(tpb, 7) == 2
        @test Angular.leftindex(tpb, 24) == 4
        @test Angular.rightindex(tpb, 24) == 6

        op = Angular.ProductExtensionOperator(tpb, op1, op2)
        @test ishermitian(Angular.matrix(op))
        @test Angular.apply(7, op, 7) ≈ -1.5 * countmatrix(3)
        @test Angular.apply(7, op, 24) ≈ 0

        # tensor sums
        tsb = Angular.TensorSumBasis(b1, b2)
        @test length(tsb) == length(b1) + length(b2)

        op = Angular.SumExtensionOperator(tsb, op1, op2)
        @test ishermitian(Angular.matrix(op))
        @test Angular.apply(4, op, 5) ≈ 0.0
        @test Angular.apply(4, op, 4) ≈ countmatrix(4)
        @test Angular.apply(5, op, 5) ≈ -2.5
    end

    @testset "Clebsch-Gordan coefficients" begin
        j1, j2 = 2, 3//2
        b1, b2 = Angular.AngularBasis(j1), Angular.AngularBasis(j2)
        id1, id2 = Angular.IdentityOperator(b1), Angular.IdentityOperator(b2)
        J1, J2 = Angular.JOperatorSet(b1), Angular.JOperatorSet(b2)
        tpb = Angular.TensorProductBasis(b1, b2)
        Jz = Angular.ProductExtensionOperator(tpb, J1.Jz, id2) + Angular.ProductExtensionOperator(tpb, id1, J2.Jz)
        J₊ = Angular.ProductExtensionOperator(tpb, J1.J₊, id2) + Angular.ProductExtensionOperator(tpb, id1, J2.J₊)
        J₋ = Angular.ProductExtensionOperator(tpb, J1.J₋, id2) + Angular.ProductExtensionOperator(tpb, id1, J2.J₋)
        J = Angular.JOperatorSet(J₊, J₋, Jz)

        e = eigen(Angular.matrix(Angular.J2(J)))
        js = Angular.findj.(e.values)
        @test maximum(js) == 7//2
        @test minimum(js) == 1//2

        e = Angular.simeigen(Angular.matrix(Angular.J2(J)), Angular.matrix(J.Jz))
        for i = 1:length(tpb)
            j, m = Angular.findj(e.values[1,i]), HalfInteger(round(Int, 2*real(e.values[2,i])), 2)
            cs_ref = map(1:length(tpb)) do k
                k1, k2 = Angular.leftindex(tpb, k), Angular.rightindex(tpb, k)
                m1, m2 = Angular.m(b1, k1), Angular.m(b2, k2)
                WignerSymbols.clebschgordan(j1, m1, j2, m2, j, m)
            end
            @test abs(cs_ref' * e.vectors[:,i]) ≈ 1.0 atol=1e-10
            for k = 1:length(tpb)
                @test abs(e.vectors[k,i]) ≈ abs(cs_ref[k]) atol=1e-10
            end
        end
    end

    @testset "ls coupling" begin
        function _test_ls(ℓ)
            lsb = Angular.LSBasis(ℓ)
            @test length(lsb) == 2*(2*ℓ + 1)

            ls2j = Angular.LStoJTransform(lsb)
            B = Angular.matrix(ls2j)
            for i = 1:length(lsb), j = 1:length(lsb)
                @test B[i, j] ≈ Angular.transform(i, ls2j, j)
            end

            # Solve for the LS-to-J transformation via diagonalization
            L = Angular.JOperatorSet(Angular.LOperator, lsb)
            S = Angular.JOperatorSet(Angular.SOperator, lsb)
            J = Angular.JOperatorSet(+, L, S)
            J2, Jz = Angular.J2(J), J.Jz
            e = Angular.simeigen(Angular.matrix(J2), Angular.matrix(Jz))
            @test all(imag.(e.values) .≈ 0.0)
            js, ms = Angular.findj.(real.(e.values[1,:])), round.(HalfInteger, real.(e.values[2,:]))
            @test all(e.values[1,:] .≈ convert.(Float64, js.*(js .+ 1)))
            @test all(e.values[2,:] .≈ convert.(Float64, ms))
            let njs = sort(Angular.countunique(js))
                @test length(njs) == 2
                @test njs[1] == ((ℓ - 1//2) => 2ℓ)
                @test njs[2] == ((ℓ + 1//2) => 2ℓ + 2)
            end

            for idx = 1:size(e.values, 2)
                j, m = js[idx], ms[idx]
                i = convert(Int, (j < ℓ ? 0 : 2j-1) + (1 + m + j))
                # agreement with Glebch-Gordan coefficients up to a global phase
                @test abs.(e.vectors[:, idx]) ≈ abs.(B[i, :])
            end
        end
        _test_ls(1)
        _test_ls(2)
        _test_ls(5)
    end

    @testset "linear algebra" begin
        @test Angular.findconnected([:a, :b, :a, :c, :b, :a]) == [[1,3,6],[2,5],[4]]
        @test Angular.findconnected([1, 11, 0, 21, 12, 2], (x,y) -> abs(y-x) <= 1) == [[1,3,6],[2,5],[4]]
        @test Angular.findconnected([1, 11, 0, 21, 12, 3], (x,y) -> abs(y-x) <= 1) != [[1,3,6],[2,5],[4]]

        @test Angular.findsimilar([1.0, 1.0, 2.0]) == [[1,2] => 1.0, [3] => 2.0]
    end

    @testset "combinatorics" begin
        using Angular: combinationrank, combinationunrank
        for (i, x) in enumerate(1:5)
            @test combinationrank(x) == i
            @test combinationunrank(1, i) == (x,)
        end
        for (i, xs) in enumerate([(1,2),(1,3),(2,3)])
            @test combinationrank(xs...) == i
            @test combinationunrank(2, i) == xs
        end
        for (i, xs) in enumerate([
                (1,2,3), (1,2,4), (1,3,4), (2,3,4), (1,2,5), (1,3,5), (2,3,5)
            ])
            @test combinationrank(xs...) == i
            @test combinationunrank(3, i) == xs
        end
    end

    @testset "fock spaces" begin
        for dim = 2:4, N = 1:dim
            b = TestBasis(dim)
            fb = Angular.FermionSpace(b, N)
            @test length(fb) == binomial(dim, N)
        end

        b = TestBasis(5)
        fb = Angular.FermionSpace(b, 3)
        @test length(fb) == 10

        op = TestOperator(b)
        fop = Angular.Fermion1POperator(fb, op)
        let i = combinationrank(1, 2, 3)
            @test Angular.apply(i, fop, i) == sum(countmatrix(k) for k=1:3)
        end
        let i = combinationrank(1, 3, 5)
            @test Angular.apply(i, fop, i) == sum(countmatrix(k) for k=1:2:5)
        end
        @test Angular.apply(1, fop, 2) ≈ countmatrix(3, 4) # <1,2,3| O |1,2,4>
        @test Angular.apply(1, fop, 10) ≈ 0
    end

    @testset "LS terms" begin
        function _test_ls_term(ℓ, N)
            lsb = Angular.LSBasis(ℓ)
            @test length(lsb) == 2*(2*ℓ + 1)
            lsfb = Angular.FermionSpace(lsb, N)

            L1 = Angular.JOperatorSet(Angular.LOperator, lsb)
            S1 = Angular.JOperatorSet(Angular.SOperator, lsb)
            L = Angular.JOperatorSet(J -> Angular.Fermion1POperator(lsfb, J), L1)
            S = Angular.JOperatorSet(J -> Angular.Fermion1POperator(lsfb, J), S1)
            J = Angular.JOperatorSet(+, L, S)
            e = Angular.simeigen(
                Angular.matrix(Angular.J2(L)),
                Angular.matrix(Angular.J2(S)),
                Angular.matrix(L.Jz),
                Angular.matrix(S.Jz),
            )
            evalues = [Angular.findj.(e.values[1:2,:]); round.(HalfInteger, real.(e.values[3:4,:]))]

            ts = Angular.countunique(Term(evalues[1,i], evalues[2,i], 1) for i=1:size(evalues, 2)) |> sort
            ts_ref = Angular.countunique(terms(Orbital(:n, ℓ), N)) |> sort
            @test length(ts) == length(ts_ref)
            for ((t, n), (t_ref, ref_multiplicity)) in zip(ts, ts_ref)
                # ref_multiplicity: the number of times this term appears
                @test n == weight(t) * ref_multiplicity
            end
        end

        _test_ls_term(1, 1)
        _test_ls_term(1, 2)
        _test_ls_term(1, 3)
        _test_ls_term(2, 2)
        #_test_ls_term(2, 3) # too slow
    end

    @testset "jj terms" begin
        function _test_jj_term(j, N)
            b = Angular.AngularBasis(j)
            @test length(b) == 2*j + 1
            fb = Angular.FermionSpace(b, N)

            J1 = Angular.JOperatorSet(Angular.JOperator, b)
            J = Angular.JOperatorSet(J -> Angular.Fermion1POperator(fb, J), J1)
            e = Angular.simeigen(
                Angular.matrix(Angular.J2(J)),
                Angular.matrix(J.Jz),
            )
            evalues = [Angular.findj.(e.values[1,:])'; round.(HalfInteger, real.(e.values[2,:]))']

            ts = Angular.countunique(evalues[1,i] for i=1:size(evalues, 2)) |> sort
            ts_ref = Angular.countunique(_terms_jw(halfint(j), N)) |> sort
            @test length(ts) == length(ts_ref)
            for ((J, n), (J_ref, ref_multiplicity)) in zip(ts, ts_ref)
                # ref_multiplicity: the number of times this term appears
                @test J == J_ref
                @test n == (2*J + 1) * ref_multiplicity
            end
        end

         _test_jj_term(1//2, 1)
         _test_jj_term(1//2, 2)
         _test_jj_term(1, 1)
         _test_jj_term(1, 2)
         _test_jj_term(1, 3)
         _test_jj_term(3//2, 3)
         _test_jj_term(2, 2)
         _test_jj_term(5//2, 3)
    end

    @testset "many-particle LSJ-to-JJ" begin
        function _test_ls_fockspace(ℓ, N)
            lsb = Angular.LSBasis(ℓ)
            @test length(lsb) == 2*(2*ℓ + 1)
            ls2j = Angular.LStoJTransform(lsb)
            fst = Angular.FermionSpaceTransform(ls2j, N)
            B = Angular.matrix(fst)
            @test all(sum(abs2, B; dims=1) .≈ 1.0)

            # TODO: actually check the values..
        end
        _test_ls_fockspace(1, 1)
        _test_ls_fockspace(1, 2)
        _test_ls_fockspace(1, 3)
        _test_ls_fockspace(1, 4)
        _test_ls_fockspace(1, 5)
        _test_ls_fockspace(1, 6)
        _test_ls_fockspace(2, 1)
        _test_ls_fockspace(2, 2)
        _test_ls_fockspace(2, 3)
        _test_ls_fockspace(2, 4)
        _test_ls_fockspace(3, 3)
    end
end
