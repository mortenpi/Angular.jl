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

# Basis transformations between N-fermion spaces
struct FermionSpaceTransform{N, B1 <: BasisSet, B2 <: BasisSet, BT <: BasisTransformation{B1,B2}} <: BasisTransformation{FermionSpace{N,B1},FermionSpace{N,B2}}
    bt :: BT
    fb1 :: FermionSpace{N,B1}
    fb2 :: FermionSpace{N,B2}
    function FermionSpaceTransform(
        bt :: BT,
        fb1 :: FermionSpace{N, B1},
        fb2 :: FermionSpace{N, B2}
    ) where {N, B1 <: BasisSet, B2 <: BasisSet, BT <: BasisTransformation{B1,B2}}
        @assert length(fb1) == length(fb2)
        new{N,B1,B2,BT}(bt, fb1, fb2)
    end
end
function FermionSpaceTransform(bt::BasisTransformation, N::Integer)
    FermionSpaceTransform(bt, FermionSpace(basisfrom(bt), N), FermionSpace(basisto(bt), N))
end
Base.length(fsbt::FermionSpaceTransform) = length(fsbt.fb1)
basisfrom(fsbt::FermionSpaceTransform) = fsbt.fb1
basisto(fsbt::FermionSpaceTransform) = fsbt.fb2
function transform(i::Integer, fsbt::FermionSpaceTransform{N}, j::Integer) where N
    is, js = combinationunrank(N, i), combinationunrank(N, j)
    sum(prod(transform(is[perm[q]], fsbt.bt, js[q]) for q = 1:N) * levicivita(perm) for perm in permutations(1:N))
end
