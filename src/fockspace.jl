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

struct Fermion1POperator{N, B <: BasisSet, O <: LinearOperator{B}} <: LinearOperator{FermionSpace{N,B}}
    b :: FermionSpace{N,B}
    op :: O
    function Fermion1POperator(fs::FermionSpace{N,B}, op::O) where {N, B <: BasisSet, O <: LinearOperator{B}}
        new{N,B,O}(fs, op)
    end
end
basis(op::Fermion1POperator) = op.b

function apply(i::Integer, op::Fermion1POperator, j::Integer)
    N = nparticles(basis(op))
    is, js =  combinationunrank(N, i),  combinationunrank(N, j)
    # TODO: the implementation can actually be optimized by noting that the matrix element is
    # non-zero only if is == js or they differ by one element.
    mij :: ComplexF64 = 0.0
    for ni = 1:N, nj = 1:N
        _is = (is[1:ni-1]..., is[ni+1:end]...)
        _js = (js[1:nj-1]..., js[nj+1:end]...)
        _is == _js || continue
        mij += (-1)^(ni + nj) * apply(is[ni], op.op, js[nj])
    end
    return mij
end
