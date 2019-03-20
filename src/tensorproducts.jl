# Tensor product basis
# ------------------------------------------------------------------------------------------
struct TensorProductBasis{B1 <: BasisSet, B2 <: BasisSet} <: BasisSet
    b1 :: B1
    b2 :: B2
end
Base.length(b::TensorProductBasis) = length(b.b1) * length(b.b2)
leftindex(b::TensorProductBasis, idx::Integer) = rem(idx - 1, length(b.b1)) + 1
rightindex(b::TensorProductBasis, idx::Integer) = div(idx - 1, length(b.b1)) + 1

struct ProductExtensionOperator{B1 <: BasisSet, B2 <: BasisSet, O1 <: LinearOperator{B1}, O2 <: LinearOperator{B2}} <: LinearOperator{TensorProductBasis{B1, B2}}
    tpb :: TensorProductBasis{B1, B2}
    op1 :: O1
    op2 :: O2
end
basis(op::ProductExtensionOperator) = op.tpb
function apply(i::Integer, op::ProductExtensionOperator, j::Integer)
    i1, i2 = leftindex(op.tpb, i), rightindex(op.tpb, i)
    j1, j2 = leftindex(op.tpb, j), rightindex(op.tpb, j)
    apply(i1, op.op1, j1) * apply(i2, op.op2, j2)
end

# Tensor sum basis
# ------------------------------------------------------------------------------------------
struct TensorSumBasis{B1 <: BasisSet, B2 <: BasisSet} <: BasisSet
    b1 :: B1
    b2 :: B2
end
Base.length(b::TensorSumBasis) = length(b.b1) + length(b.b2)

struct SumExtensionOperator{B1 <: BasisSet,B2 <: BasisSet,O1 <: LinearOperator{B1}, O2 <: LinearOperator{B2}} <: LinearOperator{TensorSumBasis{B1, B2}}
    tsb :: TensorSumBasis{B1, B2}
    op1 :: O1
    op2 :: O2
end
basis(op::SumExtensionOperator) = op.tsb
function apply(i::Integer, op::SumExtensionOperator, j::Integer) :: ComplexF64
    N1 = length(op.tsb.b1)
    if i <= N1 && i <= N1
        apply(i, op.op1, j)
    elseif i > N1 && i > N1
        apply(i - N1, op.op2, j - N1)
    else
        0.0
    end
end
