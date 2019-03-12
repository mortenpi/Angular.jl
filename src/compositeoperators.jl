# Standard composite operators

struct ProductOperator{B <: BasisSet, O1 <: LinearOperator{B}, O2 <: LinearOperator{B}} <: LinearOperator{B}
    op1 :: O1
    op2 :: O2

    function ProductOperator(o1 :: LinearOperator, o2 :: LinearOperator)
        @assert basis(o1) == basis(o2)
        B = typeof(basis(o1))
        new{B,typeof(o1),typeof(o2)}(o1, o2)
    end
end
basis(op::ProductOperator) = basis(op.op1)
apply(i::Integer, op::ProductOperator, j::Integer) =
    sum(apply(i, op.op1, k)*apply(k, op.op2, j) for k = 1:length(basis(op)))

struct SumOperator{B <: BasisSet, O1 <: LinearOperator{B}, O2 <: LinearOperator{B}} <: LinearOperator{B}
    op1 :: O1
    op2 :: O2

    function SumOperator(o1 :: LinearOperator, o2 :: LinearOperator)
        @assert basis(o1) == basis(o2)
        B = typeof(basis(o1))
        new{B,typeof(o1),typeof(o2)}(o1, o2)
    end
end
basis(op::SumOperator) = basis(op.op1)
apply(i::Integer, op::SumOperator, j::Integer) = apply(i, op.op1, j) + apply(i, op.op2, j)

struct ScalarMultiplicationOperator{S <: Number, B <: BasisSet, O <: LinearOperator{B}} <: LinearOperator{B}
    α :: S
    op :: O

    function ScalarMultiplicationOperator(α :: S, op :: LinearOperator) where {S <: Number}
        B = typeof(basis(op))
        new{S,B,typeof(op)}(α, op)
    end
end
basis(op::ScalarMultiplicationOperator) = basis(op.op)
apply(i::Integer, op::ScalarMultiplicationOperator, j::Integer) = op.α * apply(i, op.op, j)

# Extending addition and multiplication
Base.:(+)(op1::LinearOperator{B}, op2::LinearOperator{B}) where {B <: BasisSet} = SumOperator(op1, op2)
Base.:(-)(op::LinearOperator) = ScalarMultiplicationOperator(-1, op)
Base.:(-)(op1::LinearOperator{B}, op2::LinearOperator{B}) where {B <: BasisSet} = SumOperator(op1, -op2)
Base.:(*)(op1::LinearOperator{B}, op2::LinearOperator{B}) where {B <: BasisSet} = ProductOperator(op1, op2)
Base.:(*)(α::Number, op::LinearOperator) = ScalarMultiplicationOperator(α, op)
Base.:(*)(op::LinearOperator, α::Number) = ScalarMultiplicationOperator(α, op)
Base.:(/)(op::LinearOperator, α::Number) = ScalarMultiplicationOperator(1/α, op)
function Base.:(^)(op::LinearOperator, n::Integer)
    @assert n > 0
    n == 1 && return op
    ProductOperator(op, op^(n-1))
end
