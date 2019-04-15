using AtomicLevels: Orbital, RelativisticOrbital, Configuration
angularmomentum(o::Orbital) = o.ℓ
angularmomentum(o::RelativisticOrbital) = AtomicLevels.kappa_to_j(o.κ)

struct SubshellBasis{N,O<:AbstractOrbital} <: BasisSet
    ab :: AngularBasis
    fb :: FermionSpace{N,AngularBasis}
    orbital :: O
    w :: Int

    function SubshellBasis(orbital::AbstractOrbital, w::Integer)
        1 <= w <= AtomicLevels.degeneracy(orbital) || error("Bad number of particles w=$w")
        ab = AngularBasis(angularmomentum(orbital))
        new{w,typeof(orbital)}(ab, FermionSpace(ab, w), orbital, w)
    end
end
Base.length(sb::SubshellBasis) = length(sb.fb)
function Base.:(==)(a::SubshellBasis{N,O}, b::SubshellBasis{N,O}) where {N, O <: AbstractOrbital}
    (a.orbital == b.orbital) && (a.w == b.w)
end
Base.:(==)(::SubshellBasis, ::SubshellBasis) = false

# TODO
# function ms(ssb::SubshellBasis, idx::Integer)
#     mvalues =
#     spindices(ssb.fb, idx)
# end

struct ConfigurationBasis{O<:AbstractOrbital} <: BasisSet
    os :: Vector{O}
    ws :: Vector{Int}
    bs :: Vector{SubshellBasis}

    function ConfigurationBasis(subshells :: Vector{SSB}) where {O <: AbstractOrbital, SSB <: SubshellBasis{<:Any,O}}
        os = [ssh.orbital for ssh in subshells]
        length(unique(os)) == length(os) || error("Not all orbitals are unique.")
        ws = [ssh.w for ssh in subshells]
        new{typeof(first(os))}(os, ws, subshells)
    end
end
ConfigurationBasis(a::SubshellBasis, b::SubshellBasis) = ConfigurationBasis([a, b])
ConfigurationBasis(cb::ConfigurationBasis, ssh::SubshellBasis) = ConfigurationBasis([cb.bs..., ssh])

Base.length(cb::ConfigurationBasis) = prod(length(b) for b in cb.bs)

configuration(cb::ConfigurationBasis) = Configuration(cb.os, cb.ws)

struct JJSubshellStates1{N}
    J :: HalfInteger
    cs :: Matrix{ComplexF64}
    basis :: SubshellBasis{N,RelativisticOrbital{Int}}
    jjcsfs :: Ref{JJCSFs{N}}

    # function JJSubshellStates1(
    #     basis::SubshellBasis{N,RelativisticOrbital{Int}},
    #     jjcsfs::JJCSFs{N},
    #     J::Real, M::Real, α::Union{Integer,Missing} = missing
    # ) where N
    #     @show basis.orbital
    #     AtomicLevels.kappa_to_j(basis.orbital.κ) == jjcsfs.j || error("Orbital does not match JJCSFs J")
    #     J, M = convert(HalfInteger, J), convert(HalfInteger, M)
    #     idxs = findall(jjcsfs, J, M)
    #     length(idxs) == 0 && error("Bad J/M values ($J / $M)")
    #     if ismissing(α) && length(idxs) > 1
    #         error("Must specify α: $(length(idxs)) states with same J/M")
    #     end
    #     α = ismissing(α) ? 1 : α
    #     if !(1 <= α <= length(idxs))
    #         error("Invalid α value")
    #     end
    #     state = jjcsfs[idxs[α]]
    #     @assert length(state) == length(basis)
    #     new{N}(J, M, α, basis, state.cs, jjcsfs)
    # end
    function JJSubshellStates1(
        basis::SubshellBasis{N,RelativisticOrbital{Int}},
        jjcsfs::JJCSFs{N},
        J :: Real
    ) where N
        AtomicLevels.kappa_to_j(basis.orbital.κ) == jjcsfs.j || error("Orbital does not match JJCSFs J")
        J = convert(HalfInteger, J)
        @assert length(basis) == length(jjcsfs)
        cs = Matrix{ComplexF64}(undef, length(basis), convert(Int, 2J+1))
        display(jjcsfs.values)
        for (i, M) = enumerate(-J:J)
            idxs = findall(jjcsfs, J, M)
            if length(idxs) == 0
                Js = sort(unique(jjcsfs.values[1,i] for i = 1:size(jjcsfs.values,2)))
                error("Bad J. Valid: $(Js)")
            end
            length(idxs) > 1 && error("Can't handle degenerate spaces yet. Soz.")
            state = jjcsfs[first(idxs)]
            cs[:,i] = state.cs
        end
        new{N}(J, cs, basis, jjcsfs)
    end
end

struct JJCSFStates1
    J :: HalfInteger
    basis :: ConfigurationBasis{RelativisticOrbital{Int}}
    cs :: Matrix{ComplexF64}

    # function JJCSFState3(
    #     a::JJSubshellState,
    #     b::JJSubshellState,
    #     J::Real, M::Real
    # ) where N
    #     J, M = convert(HalfInteger, J), convert(HalfInteger, M)
    #     J ∈ abs(a.J - b.J):(a.J + b.J) || error("Bad J")
    #     M ∈ -J:J || error("Bad M")
    #
    #     clebschgordan(a.J, b.J, )
    #
    #     new{N}(J, M, α, basis, state.cs, jjcsfs)
    # end
end

function couple(a::Union{JJSubshellStates1,JJCSFStates1}, b::JJSubshellStates1, J::Real)
    J = convert(HalfInteger, J)
    J ∈ abs(a.J - b.J):(a.J + b.J) || error("Bad J")
    cb = ConfigurationBasis(a.basis, b.basis)
    @assert length(a.basis) * length(b.basis) == length(cb)
    JJCSFStates1(J, cb, _couple(a.J, a.cs, b.J, b.cs, J))
end

function _couple(
        JA::HalfInteger, A::Matrix{ComplexF64},
        JB::HalfInteger,  B::Matrix{ComplexF64},
        J::HalfInteger)
    J ∈ abs(JA - JB):(JA + JB) || error("Bad J")

    @assert size(A, 2) == (2JA+1)
    @assert size(B, 2) == (2JB+1)
    N = size(A,1)*size(B,1)

    @show size(A) size(B)

    cs = zeros(ComplexF64, N, convert(Int, 2J+1))
    for (k, M) = enumerate(-J:J), (i, M1) = enumerate(-JA:JA), (j, M2) = enumerate(-JB:JB)
        cg = clebschgordan(JA, M1, JB, M2, J, M)
        for q in 1:N
            # column-major
            ib, ia = divrem(q - 1, size(A, 1)) .+ (1, 1)
            #@show cg ib, ia
            cs[q,k] += cg * A[ia, i] * B[ib, j]
        end
    end
    cs
end

# TODO:
#
#  - convert JJCSFs to State{SubshellBasis}
#  - couple JJCSFs with CG coefficients
#  -
