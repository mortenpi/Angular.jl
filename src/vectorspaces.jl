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

function apply(op::LinearOperator{B}, cs::Vector{<:Number}) where B <: BasisSet
    N = length(basis(op))
    length(cs) == N || error("cs must have $(length(basis(op))) elements")
    ComplexF64[sum(apply(i, op, j) * cs[j] for j = 1:N) for i = 1:N]
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

# States BasisSet
struct State{B <: BasisSet}
    b :: B
    cs :: Vector{ComplexF64}

    function State(b::BasisSet, cs::Vector{<:Number})
        length(b) == length(cs) || error("Must have $(length(b)) components")
        new{typeof(b)}(b, cs)
    end
end
Base.length(v::State) = length(v.cs)
basis(s::State) = s.b
function Base.:(+)(v1::State{B}, v2::State{B}) where {B <: BasisSet}
    v1.b == v2.b || error("Basis sets must match")
    State(v1.b, v1.cs .+ v2.cs)
end
Base.:(*)(c::Number, v::State) = State(basis(v), c .* v.cs)
LinearAlgebra.normalize(v::State) = State(basis(v), v.cs ./ sqrt(sum(abs2.(v.cs))))
