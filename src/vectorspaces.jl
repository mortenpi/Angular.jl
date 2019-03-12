"""
    abstract type BasisSet

An abstract type that represents a set of orthonormal states that can be used to generate
a vector space.

The basis sets are assumed to be indexed with indices in `1:length(b)`.

# Interface

Child types must implement `Base.length`, which should return the number of basis elements
(or, equivalently, the dimension of the generated vector space).
"""
abstract type BasisSet end

"""
    abstract type LinearOperator{B <: BasisSet}

# Interface

Where `L <: LinearOperator` is the user-defined linear operator type:

* `apply(op::L, idx::Integer)`: return a `VSVector{B}` corresponding to the coefficients
  of `op` applied on the `idx` basis state
* `basis(op::L)`: return the basis set
"""
abstract type LinearOperator{B <: BasisSet} end

function apply end
function basis end

struct VSVector{B <: BasisSet}
    cs :: Vector{ComplexF64}
end
Base.length(v::VSVector) = length(v.cs)
Base.:(+)(v1::VSVector{B}, v2::VSVector{B}) where {B <: BasisSet} = VSVector{B}(v1.cs .+ v2.cs)

function apply(op::LinearOperator{B}, x::VSVector{B}) where B <: BasisSet
    v = zeros(ComplexF64, length(basis(op)))
    for i in 1:1:length(basis(op))
        Lei = apply(op, i)
        v .+= x.cs[i] .* Lei.cs
    end
    return VSVector{B}(v)
end

function apply(op::LinearOperator{B}, cs::Vector) where B <: BasisSet
    @assert length(cs) == length(basis(op))
    apply(op, VSVector{B}(cs))
end

function matrixelement(op::LinearOperator, i::Integer, j::Integer)
    vj = apply(op, j)
    vj.cs[i]
end

"""
    matrix(op::LinearOperator)

Returns a matrix representation of the operator `op`.
"""
function matrix(op::LinearOperator)
    N = length(basis(op))
    [matrixelement(op, i, j) for i = 1:N, j = 1:N]
end
