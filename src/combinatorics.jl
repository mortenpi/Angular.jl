"""
    combinationrank(xs...)

Returns the index of the combination `xs` in the combinatorial number system. The elements
of `xs` all have to be unique, greater than zero and sorted in ascending order.
"""
function combinationrank(xs...)
    @assert length(unique(xs)) == length(xs)
    @assert sort([xs...]) == [xs...]
    @assert all(xs .> 0)
    k = length(xs)
    1 + sum(binomial.(xs .- 1, 1:k))
end

"""
    combinationunrank(k, i)

Returns the `i`-th `k`-element combination in the combinatorial number system.
"""
function combinationunrank(k, idx)
    @assert k > 0
    @assert idx > 0
    v = zeros(Int, k)
    for i = k:-1:1
        ck = _max_ck(i, idx - 1)
        v[i] = ck + 1
        idx -= binomial(ck, i)
    end
    tuple(v...)
end

function _max_ck(k, idx)
    ck = 0
    while true
        binomial(ck + 1, k) > idx && return ck
        ck += 1
    end
    error("Unreachable reached.")
end

"""
    findcombinationdiff(c1::NTuple{N,Int}, c2::NTuple{N,Int}) -> (idx1, idx2) or nothing

If the two ordered combinations of the same length have all elements except one in common,
return a pair of indices of those elements. Otherwise, return `nothing`.
"""
function findcombinationdiff(c1::NTuple{N,Int}, c2::NTuple{N,Int}) where N
    idx1::Union{Nothing,Int}, idx2::Union{Nothing,Int} = nothing, nothing
    i = 1
    while i < N
        i1 = isnothing(idx1) ? i : i + 1
        i2 = isnothing(idx2) ? i : i + 1
        if c1[i1] == c2[i2]
            i += 1
        elseif (c1[i1] < c2[i2]) && isnothing(idx1)
            idx1 = i1
        elseif (c1[i1] > c2[i2]) && isnothing(idx2)
            idx2 = i2
        else
            return nothing
        end
    end
    isnothing(idx1) && (idx1 = N)
    isnothing(idx2) && (idx2 = N)
    c1[idx1] == c2[idx2] ? nothing : (idx1, idx2)
end
