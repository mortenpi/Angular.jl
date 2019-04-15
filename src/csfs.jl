struct LSCSFs{N}
    ℓ :: Int
    w :: Int
    terms :: Vector{Tuple{HalfInteger,HalfInteger}} # TODO: remove this
    lsb :: LSBasis
    fb ::FermionSpace{N,LSBasis}
    values :: Matrix{HalfInteger}
    states :: Matrix{ComplexF64}

    function LSCSFs(ℓ, w, fb :: FermionSpace{N,LSBasis}, values::Matrix, states::Matrix) where N
        n = length(fb)
        size(states) == (n, n) || throw(DomainError(size(states), "Wrong size for CSF coef. matrix"))
        size(values) == (4, n) || throw(DomainError(size(values), "Wrong size for eigenvalue matrix"))
        values::Matrix{HalfInteger} = convert.(HalfInteger, values)
        # pairing as (S, L) so that sorting would match Terms
        lspairs = sort(countunique((values[2,i],values[1,i]) for i = 1:n))
        terms = sort(countunique(AtomicLevels.terms(AtomicLevels.Orbital(:x, ℓ), w)))
        for ((ls, count_ls), (t, count_t)) in zip(lspairs, terms)
            S, L = ls
            count_ls == AtomicLevels.weight(t) * count_t || throw(DomainError("Bad eigenvalue list provided"))
            mspairs = sort!([
                (values[3, i], values[4, i])
                for i in findall(idx -> values[1,idx] == L && values[2,idx] == S, 1:size(values, 2))
            ])
            mspairs_ref = sort(repeat([(ML, MS) for ML=-L:L for MS=-S:S], count_t))
            if !(mspairs == mspairs_ref)
                @error "mspairs != mspairs_ref" mspairs mspairs_ref
                throw(DomainError("Bad eigenvalue list provided"))
            end
        end
        new{N}(ℓ, w, [t for (t, _) in lspairs], spbasis(fb), fb, values, states)
    end
end
Base.length(csfs::LSCSFs) = length(csfs.fb)

function Base.getindex(csfs::LSCSFs, idx::Integer)
    1 <= idx <= length(csfs) || throw(BoundsError(csfs, idx))
    L, S, ML, MS = csfs.values[1, idx], csfs.values[2, idx], csfs.values[3, idx], csfs.values[4, idx]
    LSState(csfs, L, S, ML, MS, csfs.states[:, idx])
end

function lscsfs(ℓ::Integer, w::Integer)
    lsb = LSBasis(ℓ)
    lsfb = FermionSpace(lsb, w)
    L1 = JOperatorSet(LOperator, lsb)
    S1 = JOperatorSet(SOperator, lsb)
    L = JOperatorSet(J -> Fermion1POperator(lsfb, J), L1)
    S = JOperatorSet(J -> Fermion1POperator(lsfb, J), S1)
    J = JOperatorSet(+, L, S)
    e = simeigen(matrix(J2(L)), matrix(J2(S)), matrix(L.Jz), matrix(S.Jz),)
    evalues = [findj.(e.values[1:2,:]); round.(HalfInteger, real.(e.values[3:4,:]))]
    #display(evalues)
    LSCSFs(ℓ, w, lsfb, evalues, e.vectors)
end

struct LSState{N}
    csfs :: LSCSFs{N}
    L :: HalfInteger
    S :: HalfInteger
    ML :: HalfInteger
    MS :: HalfInteger
    cs :: Vector{ComplexF64}
end
Base.length(ls::LSState) = length(ls.cs)

function Base.show(io::IO, ls::LSState)
    lsb = spbasis(ls.csfs.fb)
    n, ℓ, w = length(lsb), Angular.ℓ(lsb), nparticles(ls.csfs.fb)
    println(io, "LS state |$(ℓ)^$(w); $(ls.L),$(ls.S),$(ls.ML),$(ls.MS)> =")
    mls = [convert(Int, ml(lsb, i)) for i=1:n]
    mss = [ms(lsb, i) > 0 ? "↑" : "↓" for i=1:n]
    mlabels = [format("({:3d},{})", ml, ms) for (ml, ms) in zip(mls, mss)]
    isfirst = true
    for i in 1:length(ls)
        idxs = spindices(ls.csfs.fb, i)
        lbl = join(map(idx -> mlabels[idx], idxs), " ")
        c = ls.cs[i]
        @assert imag(c) ≈ 0
        if !(abs(real(c)) < 1e-10 )
            printfmtln(io, "{} {:15:8f} |{}>", (isfirst ? " " : "+"), real(c), lbl)
            isfirst = false
        end
    end
end

# jj-coupled CSFs
struct JJCSFs{N}
    j :: HalfInteger
    w :: Int
    terms :: Vector{HalfInteger}
    lsb :: AngularBasis
    fb ::FermionSpace{N,AngularBasis}
    values :: Matrix{HalfInteger}
    states :: Matrix{ComplexF64}

    function JJCSFs(j, w, fb :: FermionSpace{N,AngularBasis}, values::Matrix, states::Matrix) where N
        n = length(fb)
        size(states) == (n, n) || throw(DomainError(size(states), "Wrong size for CSF coef. matrix"))
        size(values) == (2, n) || throw(DomainError(size(values), "Wrong size for eigenvalue matrix"))
        values::Matrix{HalfInteger} = convert.(HalfInteger, values)
        Js = [values[1,i] for i = 1:n]
        # for (J, count) in sort(countunique(Js))
        #     rem(count, convert(Int, 2*J + 1)) == 0 || throw(DomainError(values, "Bad eigenvalue list provided"))
        #     mspairs = sort!([values[2, i] for i in findall(isequal(J), Js)])
        #     mspairs_ref = [M for M=-J:J]
        #     if !(mspairs == mspairs_ref)
        #         @error "mspairs != mspairs_ref" mspairs mspairs_ref
        #         throw(DomainError("Bad eigenvalue list provided"))
        #     end
        # end
        new{N}(j, w, unique(Js), spbasis(fb), fb, values, states)
    end
end
Base.length(csfs::JJCSFs) = length(csfs.fb)

function jjcsfs(j::Real, w::Integer)
    j::HalfInteger = convert(HalfInteger, j)
    b = AngularBasis(j)
    fb = FermionSpace(b, w)
    J1 = JOperatorSet(JOperator, b)
    J = JOperatorSet(J -> Fermion1POperator(fb, J), J1)
    e = simeigen(matrix(J2(J)), matrix(J.Jz),)
    evalues = [findj.(e.values[1,:])'; round.(HalfInteger, real.(e.values[2,:]))']
    JJCSFs(j, w, fb, evalues, e.vectors)
end

function Base.getindex(csfs::JJCSFs, idx::Integer)
    1 <= idx <= length(csfs) || throw(BoundsError(csfs, idx))
    J, M = csfs.values[1, idx], csfs.values[2, idx]
    JJState(csfs, J, M, csfs.states[:, idx])
end

function Base.findall(jjcsfs::JJCSFs, J::Real, M::Real)
    J, M = convert(HalfInteger, J), convert(HalfInteger, M)
    idxs = Int[]
    for i = 1:size(jjcsfs.values, 2)
        jjcsfs.values[1,i] == J && jjcsfs.values[2,i] == M && push!(idxs, i)
    end
    return idxs
end

struct JJState{N}
    csfs :: JJCSFs{N}
    J :: HalfInteger
    M :: HalfInteger
    cs :: Vector{ComplexF64}
end
Base.length(jjs::JJState) = length(jjs.cs)

function Base.show(io::IO, jjs::JJState)
    b = spbasis(jjs.csfs.fb)
    n, j, w = length(b), b.j, nparticles(jjs.csfs.fb)
    println(io, "jj state |[$(j)]^$(w); $(jjs.J),$(jjs.M)> =")
    mlabels = [format("({:>5s})", m(b, i)) for i in 1:n]
    isfirst = true
    for i in 1:length(jjs)
        idxs = spindices(jjs.csfs.fb, i)
        lbl = join(map(idx -> mlabels[idx], idxs), " ")
        c = jjs.cs[i]
        @assert imag(c) ≈ 0
        if !(abs(real(c)) < 1e-10 )
            printfmtln(io, "{} {:15:8f} |{}>", (isfirst ? " " : "+"), real(c), lbl)
            isfirst = false
        end
    end
end
