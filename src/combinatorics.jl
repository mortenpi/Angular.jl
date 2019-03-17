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
