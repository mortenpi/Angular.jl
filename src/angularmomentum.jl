"""
    struct AngularBasis <: BasisSet

Defines a set of states with angular momentum `j`. The dimensionality if `2j+1`.

# Interface

* `m(b::AngularBasis, i)`: returns the ``m`` quantum number of the `i`th state.
"""
struct AngularBasis <: BasisSet
    j :: HalfInteger
    function AngularBasis(j)
        @assert j >= 0
        new(j)
    end
end
Base.length(b::AngularBasis) = convert(Int, 2 * b.j + 1)
m(b::AngularBasis, idx) = -b.j + (idx - 1)

"""
    struct JOperator{K} <: LinearOperator{AngularBasis}

Type to represent the angular momentum operators ``J_z``, ``J_+`` and ``J_-``, acting on
states with defined total angular momentum ([`AngularBasis`](@ref)).
"""
struct JOperator{K} <: LinearOperator{AngularBasis}
    b :: AngularBasis
end
function JOperator(b::AngularBasis, kind::Symbol)
    kind in [:z, :+, :-] || throw(ArgumentError("Bad kind: $kind"))
    JOperator{kind}(b)
end
basis(op::JOperator) = op.b
function apply(i::Integer, op::JOperator{K}, j::Integer) :: ComplexF64 where K
    angJ = convert(ComplexF64, op.b.j)
    if K == :z
        (i == j) ? m(op.b, j) : 0.0
    elseif K == :+
        (i == j + 1) ? sqrt((angJ - m(op.b, j)) * (angJ + m(op.b, j) + 1)) : 0.0
    elseif K == :-
        (i == j - 1) ? sqrt((angJ + m(op.b, j)) * (angJ - m(op.b, j) + 1)) : 0.0
    else
        error("Unreachable reached.")
    end
end

struct JOperatorSet{T,U,V}
    J₊ :: T
    J₋ :: U
    Jz :: V
end
function JOperatorSet(b::BasisSet)
    Jz = JOperator(b, :z)
    J₊, J₋ = JOperator(b, :+), JOperator(b, :-)
    JOperatorSet(J₊, J₋, Jz)
end

Jx(op::JOperatorSet) = (op.J₊ + op.J₋) / 2
Jy(op::JOperatorSet) = (op.J₊ - op.J₋) / (2im)
J2(op::JOperatorSet) = Jx(op)^2 + Jy(op)^2 + op.Jz^2

"""
    findj(λ) -> HalfInteger

Convert and ``J^2`` operator eigenvalue into the corresponding half-integer ``j`` value.
"""
findj(λ) = HalfInteger(isqrt(1 + round(Int, real(4λ))) - 1, 2)
