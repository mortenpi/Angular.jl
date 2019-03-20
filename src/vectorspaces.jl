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

* `apply(i::Integer, op::L, j::Integer)`: return the `i`-th compoment of the vector that results
  from applying operator `op` on the basis state `j`.
* `basis(op::L)`: return the basis set
"""
abstract type LinearOperator{B <: BasisSet} end

"""
    abstract type BasisTransformation{B1 <: BasisSet, B2 <: BasisSet}

Abstract type for representations of basis transformations between two basis sets, from `B1` to `B2`.

# Interface

* `basisfrom(::BasisTransformation)`: return the basis `B1`
* `basisto(::BasisTransformation)`: return the basis `B2`
* `transform(i::Integer, ::BasisTransformation, j::Integer)`: return the `i`-th component of the vector
  in `B2` corresponding to the transformed `j`-th vector of `B1`.
"""
abstract type BasisTransformation{B1 <: BasisSet, B2 <: BasisSet} end

function apply end
function basis end
function basisfrom end
function basisto end
function transform end

struct VSVector{B <: BasisSet}
    cs :: Vector{ComplexF64}
end
Base.length(v::VSVector) = length(v.cs)
Base.:(+)(v1::VSVector{B}, v2::VSVector{B}) where {B <: BasisSet} = VSVector{B}(v1.cs .+ v2.cs)

function apply(op::LinearOperator{B}, x::VSVector{B}) where B <: BasisSet
    N = length(basis(op))
    VSVector{B}(ComplexF64[
        sum(apply(i, op, j) * x.cs[j] for j = 1:N)
        for i = 1:N
    ])
end

function apply(op::LinearOperator{B}, cs::Vector) where B <: BasisSet
    @assert length(cs) == length(basis(op))
    apply(op, VSVector{B}(cs))
end

"""
    matrix(op::LinearOperator)

Returns a matrix representation of the operator `op`.
"""
function matrix(op::LinearOperator)
    N = length(basis(op))
    [apply(i, op, j) for i = 1:N, j = 1:N]
end

"""
    matrix(bt::BasisTransformation)

Return the matrix representation of the basis transformation `bt`.
"""
function matrix(bt::BasisTransformation)
    N1, N2 = length(basisfrom(bt)), length(basisto(bt))
    [transform(i, bt, j) for i = 1:N2, j = 1:N1]
end

struct IdentityOperator{B <: BasisSet} <: LinearOperator{B}
    b :: B
end
basis(op::IdentityOperator) = op.b
apply(i::Integer, op::IdentityOperator, j::Integer) = ComplexF64(i == j ? 1.0 : 0.0)
