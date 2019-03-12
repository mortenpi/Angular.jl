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
function Angular.apply(op::TestOperator, idx::Integer)
    v = zeros(ComplexF64, length(op.b))
    v[idx] = op.diag[idx]
    Angular.VSVector{TestBasis}(v)
end
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
    end
end
