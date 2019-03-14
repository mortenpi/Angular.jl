const NaNCF64 = (NaN+NaN*im) :: ComplexF64

function findconnected(xs::Vector, cmp=isequal)
    # NOTE: `cmp` must be symmetric.
    components = Vector{Int}[]
    visited = fill(false, length(xs))
    for i = 1:length(xs)
        visited[i] && continue # the value is already part of a component
        component = Int[]
        _find_component!(component, xs, visited, cmp, i)
        push!(components, component)
    end
    return components
end

function _find_component!(component, xs, visited, cmp, i)
    push!(component, i)
    visited[i] = true
    for j = (i+1):length(xs)
        visited[j] && continue # the value is already part of a component
        cmp(xs[i], xs[j]) && _find_component!(component, xs, visited, cmp, j)
    end
end

function findsimilar(xs::Vector{T}; atol=sqrt(eps(T))) where T <: Number
    ccs = findconnected(xs, (x,y) -> isapprox(x, y; atol=atol))
    map(ccs) do idxs
        idxs => mean(xs[idxs])
    end
end

countunique(xs) = [uq => count(isequal(uq), xs) for uq in unique(xs)]
findunique(xs) = [uq => findall(isequal(uq), xs) for uq in unique(xs)]

function subspace_matrix(M::AbstractMatrix, vectors::AbstractMatrix, subspace::Vector{Int})
    vs = vectors[:, subspace]
    # NOTE: also computing the transpose seems to give some numeric stability
    (vs' * M * vs + (vs' * M' * vs)')/2
end

function simeigen(A::AbstractMatrix, Bs...; atol=1e-10)
    for B in Bs
        @assert typeof(B) <: AbstractMatrix
        @assert size(A) == size(B)
    end

    e = eigen(A)
    if length(Bs) == 0
        return (values = e.values, vectors = e.vectors)
    else
        values = fill(NaNCF64, (length(Bs) + 1, length(e.values)))
        vectors = fill(NaNCF64, size(e.vectors))
        subspaces = findsimilar(e.values; atol=atol)
        for (ids, ev) in subspaces
            Bs_ss = [subspace_matrix(B, e.vectors, ids) for B in Bs]
            simeig = simeigen(Bs_ss...; atol=atol)
            values[1, ids] .= ev
            values[2:end, ids] = simeig.values
            vectors[:,ids] = e.vectors[:,ids] * simeig.vectors
        end
        return (values = values, vectors = vectors)
    end
end
