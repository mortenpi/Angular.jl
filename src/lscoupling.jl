"""
    struct LSBasis <: BasisSet

Represent a basis of ``L^2`` and ``S^2`` eigenstates, for an integer orbital angular
momentum ``ℓ``.
"""
struct LSBasis <: BasisSet
    b :: TensorProductBasis{AngularBasis,AngularBasis}
    function LSBasis(ℓ::Integer)
        new(TensorProductBasis(AngularBasis(ℓ), AngularBasis(1//2)))
    end
end
Base.length(b::LSBasis) = length(b.b)

ℓ(b::LSBasis) = convert(Int, b.b.b1.j)
ml(b::LSBasis, idx) = m(b.b.b1, leftindex(b.b, idx))
ms(b::LSBasis, idx) = m(b.b.b2, rightindex(b.b, idx))

struct LOperator{K} <: LinearOperator{LSBasis}
    b :: LSBasis
    op :: ProductExtensionOperator{AngularBasis,AngularBasis,JOperator{K},IdentityOperator{AngularBasis}}
    function LOperator(b::LSBasis, kind::Symbol)
        op = ProductExtensionOperator(b.b, JOperator(b.b.b1, kind), IdentityOperator(b.b.b2))
        new{kind}(b, op)
    end
end
basis(op::LOperator) = op.b
apply(i::Integer, op::LOperator, j::Integer) = apply(i, op.op, j)

struct SOperator{K} <: LinearOperator{LSBasis}
    b :: LSBasis
    op :: ProductExtensionOperator{AngularBasis,AngularBasis,IdentityOperator{AngularBasis},JOperator{K}}
    function SOperator(b::LSBasis, kind::Symbol)
        op = ProductExtensionOperator(b.b, IdentityOperator(b.b.b1), JOperator(b.b.b2, kind))
        new{kind}(b, op)
    end
end
basis(op::SOperator) = op.b
apply(i::Integer, op::SOperator, j::Integer) = apply(i, op.op, j)

# LS to J transformation
struct LStoJTransform <: BasisTransformation{LSBasis,TensorSumBasis{AngularBasis,AngularBasis}}
    lsb :: LSBasis
    jjb :: TensorSumBasis{AngularBasis,AngularBasis}
    function LStoJTransform(lsb::LSBasis)
        ℓ(lsb) == 0 && error("ℓ=0 is trivial")
        jjb = TensorSumBasis(AngularBasis(ℓ(lsb) - 1//2), AngularBasis(ℓ(lsb) + 1//2))
        @assert length(lsb) == length(jjb)
        new(lsb, jjb)
    end
end
basisfrom(ls2j::LStoJTransform) = ls2j.lsb
basisto(ls2j::LStoJTransform) = ls2j.jjb
function transform(i::Integer, ls2j::LStoJTransform, j::Integer)
    Nj1 = length(ls2j.jjb.b1)
    _j, _m = if i <= Nj1
        ls2j.jjb.b1.j, m(ls2j.jjb.b1, i)
    else
        ls2j.jjb.b2.j, m(ls2j.jjb.b2, i - Nj1)
    end
    clebschgordan(ℓ(ls2j.lsb), ml(ls2j.lsb, j), HalfInteger(1, 2), ms(ls2j.lsb, j), _j, _m)
end
