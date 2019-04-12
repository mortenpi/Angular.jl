module Angular
using Compat
using LinearAlgebra
using Statistics
using Formatting
using WignerSymbols: HalfInteger, clebschgordan
import AtomicLevels
using AtomicLevels: AbstractOrbital

# TODO: move this to WignerSymbols
Base.round(::Type{HalfInteger}, x) = HalfInteger(round(Int, 2x), 2)
Base.:*(a::HalfInteger, b::HalfInteger) = (a.numerator * b.numerator) // 4 # converts to Rational

"""
    @halfint(x) -> HalfInteger

Performs a compile-time conversion of a number literal into a `HalfInteger`. This allows
for an easy construction of `HalfInteger` values, from e.g. integers and rationals.

```jldoctest
julia> @halfint(3//2)
3/2

julia> @halfint(3//2) |> typeof
HalfInteger
```

Trying to use `@halfint` on a non-literal value will yield a compile-time error:

```jldoctest
julia> function foo(q)
           @halfint(q)
       end
ERROR: LoadError: ArgumentError: @halfint macro needs a compile-time constant as an argument
```
"""
macro halfint(x)
    value = try
        eval(x)
    catch
        throw(ArgumentError("@halfint macro needs a compile-time constant as an argument"))
    end
    convert(HalfInteger, value)
end

"""
    halfint(x) -> HalfInteger

Converts any valid numeric value to a `HalfInteger`. A shorthand for `convert(HalfInteger, x)`.
"""
halfint(x) = convert(HalfInteger, x)

include("combinatorics.jl")
include("linearalgebra.jl")
include("vectorspaces.jl")
include("compositeoperators.jl")
include("tensorproducts.jl")
include("fockspace.jl")
include("angularmomentum.jl")
include("lscoupling.jl")
include("csfs.jl")
include("atomicconfigurations.jl")

end # module
