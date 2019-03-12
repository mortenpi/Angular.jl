using Angular
using LinearAlgebra
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
        end
    end
end
