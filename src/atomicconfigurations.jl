using AtomicLevels: Orbital, RelativisticOrbital
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


# TODO:
#
#  - convert JJCSFs to State{SubshellBasis}
#  - couple JJCSFs with CG coefficients
