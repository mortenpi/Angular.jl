using Angular
using LinearAlgebra
using WignerSymbols
using WignerSymbols: HalfInteger
using Test

# ==========================================================================================
# TestBasis
# ------------------------------------------------------------------------------------------
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
    diag :: Vector{ComplexF64}
    function TestOperator(b, diag)
        @assert length(b) == length(diag)
        new(b, diag)
    end
end
Angular.basis(op::TestOperator) = op.b
Angular.apply(i::Integer, op::TestOperator, j::Integer) :: ComplexF64 = (i == j) ? op.diag[i] : 0.0
# ==========================================================================================

@testset "Angular.jl" begin
    @testset "TestBasis" begin
        b = TestBasis(3)
        @test length(b) == 3
        op = TestOperator(b, [1, 2, 3])

        v1 = Angular.apply(op, [1, 1, 1])
        v2 = Angular.apply(op, v1)
        v3 = Angular.apply(op, v1 + v2)
        @test v1.cs == [1, 2, 3]
        @test v2.cs == [1, 4, 9]
        @test v3.cs == [2, 12, 36]

        @test isdiag(Angular.matrix(op))
        @test diag(Angular.matrix(op)) == [1, 2, 3]

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
        tpb = Angular.TensorProductBasis(b1, b2)
        @test length(tpb) == length(b1) * length(b2)

        @test Angular.leftindex(tpb, 1) == 1
        @test Angular.rightindex(tpb, 1) == 1
        @test Angular.leftindex(tpb, 7) == 3
        @test Angular.rightindex(tpb, 7) == 2
        @test Angular.leftindex(tpb, 24) == 4
        @test Angular.rightindex(tpb, 24) == 6

        op1 = TestOperator(b1, [1,2,3,4])
        op2 = Angular.JOperator(b2, :z)
        op = Angular.ExtensionOperator(tpb, op1, op2)
        @test isdiag(Angular.matrix(op))
        @test Angular.apply(7, op, 7) ≈ -4.5
    end

    @testset "Clebsch-Gordan coefficients" begin
        j1, j2 = 2, 3//2
        b1, b2 = Angular.AngularBasis(j1), Angular.AngularBasis(j2)
        id1, id2 = Angular.IdentityOperator(b1), Angular.IdentityOperator(b2)
        J1, J2 = Angular.JOperatorSet(b1), Angular.JOperatorSet(b2)
        tpb = Angular.TensorProductBasis(b1, b2)
        Jz = Angular.ExtensionOperator(tpb, J1.Jz, id2) + Angular.ExtensionOperator(tpb, id1, J2.Jz)
        J₊ = Angular.ExtensionOperator(tpb, J1.J₊, id2) + Angular.ExtensionOperator(tpb, id1, J2.J₊)
        J₋ = Angular.ExtensionOperator(tpb, J1.J₋, id2) + Angular.ExtensionOperator(tpb, id1, J2.J₋)
        J = Angular.JOperatorSet(J₊, J₋, Jz)

        e = eigen(Angular.matrix(Angular.J2(J)))
        js = Angular.findj.(e.values)
        @test maximum(js) == 7//2
        @test minimum(js) == 1//2

        e = Angular.simeigen(Angular.matrix(Angular.J2(J)), Angular.matrix(J.Jz))
        @show size(e.values)
        for i = 1:length(tpb)
            j, m = Angular.findj(e.values[1,i]), HalfInteger(round(Int, 2*real(e.values[2,i])), 2)
            @show j m
            for k = 1:length(tpb)
                k1, k2 = Angular.leftindex(tpb, k), Angular.rightindex(tpb, k)
                m1, m2 = Angular.m(b1, k1), Angular.m(b2, k2)
                c = abs(e.vectors[k,i])
                c_ref = abs(WignerSymbols.clebschgordan(j1, m1, j2, m2, j, m))
                @test c ≈ c_ref atol=1e-10
            end
        end
    end

    @testset "linear algebra" begin
        @test Angular.findconnected([:a, :b, :a, :c, :b, :a]) == [[1,3,6],[2,5],[4]]
        @test Angular.findconnected([1, 11, 0, 21, 12, 2], (x,y) -> abs(y-x) <= 1) == [[1,3,6],[2,5],[4]]
        @test Angular.findconnected([1, 11, 0, 21, 12, 3], (x,y) -> abs(y-x) <= 1) != [[1,3,6],[2,5],[4]]

        @test Angular.findsimilar([1.0, 1.0, 2.0]) == [[1,2] => 1.0, [3] => 2.0]
    end
end
