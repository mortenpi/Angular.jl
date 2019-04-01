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

struct JOperatorSet{B <: BasisSet, T <: LinearOperator{B}, U <: LinearOperator{B}, V <: LinearOperator{B}}
    J₊ :: T
    J₋ :: U
    Jz :: V
end
function JOperatorSet(JOp, b::BasisSet)
    Jz = JOp(b, :z)
    J₊, J₋ = JOp(b, :+), JOp(b, :-)
    JOperatorSet{typeof(b),typeof(J₊),typeof(J₋),typeof(Jz)}(J₊, J₋, Jz)
end
JOperatorSet(b::BasisSet) = JOperatorSet(JOperator, b)

function JOperatorSet(f::Function, js::Vararg{JOperatorSet})
    length(js) > 0 || throw(ArgumentError("Must provide at least one J operator"))
    Angular.JOperatorSet(
        f((j.J₊ for j in js)...),
        f((j.J₋ for j in js)...),
        f((j.Jz for j in js)...)
    )
end

basis(op::JOperatorSet) = basis(op.Jz)

Jx(op::JOperatorSet) = (op.J₊ + op.J₋) / 2
Jy(op::JOperatorSet) = (op.J₊ - op.J₋) / (2im)
J2(op::JOperatorSet) = Jx(op)^2 + Jy(op)^2 + op.Jz^2

"""
    findj(λ) -> HalfInteger

Convert and ``J^2`` operator eigenvalue into the corresponding half-integer ``j`` value.
"""
findj(λ) = HalfInteger(isqrt(1 + round(Int, real(4λ))) - 1, 2)

struct J2Operator{B <: BasisSet, JS <: JOperatorSet{B}, EOP <: LinearOperator{B}} <: LinearOperator{B}
    J :: JS
    J2 :: EOP

    function J2Operator(jops :: JOperatorSet)
        b = basis(jops)
        J2 = (jops.J₊*jops.J₋ + jops.J₋*jops.J₊)/2 + jops.Jz^2
        new{typeof(b),typeof(jops),typeof(J2)}(jops, J2)
    end
end
basis(op::J2Operator) = basis(op.J)
apply(i::Integer, op::J2Operator, j::Integer) = apply(i, op.J2, j)
function matrix(j2op::J2Operator)
    J₊ = matrix(j2op.J.J₊)
    J₋ = matrix(j2op.J.J₋)
    Jz = matrix(j2op.J.Jz)
    (J₊*J₋ .+ J₋*J₊)./2 .+ Jz^2
end
