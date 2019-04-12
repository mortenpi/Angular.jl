"""
    struct FermionSpace{N::Int, B <: BasisSet} <: BasisSet

Represents an `N`-particle fermion space, constructed out as a `N`-fold antisymmetrized
tensor product of the single-particle basis set `B`.
"""
struct FermionSpace{N,B <: BasisSet} <: BasisSet
    b :: B

    function FermionSpace(b :: BasisSet, N :: Integer)
        length(b) >= N || error("N=$N too large for basis length(b)=$(length(b)).")
        N = convert(Int, N)
        new{N,typeof(b)}(b)
    end
end
Base.length(fs::FermionSpace{N}) where N = binomial(length(fs.b), N)
nparticles(::FermionSpace{N}) where N = N
spbasis(fb::FermionSpace) = fb.b
spindices(::FermionSpace{N}, idx) where N = combinationunrank(N, idx)

struct Fermion1POperator{N, B <: BasisSet, O <: LinearOperator{B}} <: LinearOperator{FermionSpace{N,B}}
    b :: FermionSpace{N,B}
    op :: O
    function Fermion1POperator(fs::FermionSpace{N,B}, op::O) where {N, B <: BasisSet, O <: LinearOperator{B}}
        new{N,B,O}(fs, op)
    end
end
basis(op::Fermion1POperator) = op.b

function apply(i::Integer, op::Fermion1POperator, j::Integer) :: ComplexF64
    N = nparticles(basis(op))
    if i == j
        is = combinationunrank(N, i)
        sum(apply(i, op.op, i) for i = is)
    else
        is, js =  combinationunrank(N, i),  combinationunrank(N, j)
        scd = findcombinationdiff(is, js)
        if isnothing(scd)
            0.0
        else
            idx1, idx2 = scd
            apply(is[idx1], op.op, js[idx2]) * (-1)^(idx1 + idx2)
        end
    end
end
