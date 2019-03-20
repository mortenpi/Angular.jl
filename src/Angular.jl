module Angular
using LinearAlgebra
using Statistics
using WignerSymbols: HalfInteger, clebschgordan

# TODO: move this to WignerSymbols
Base.round(::Type{HalfInteger}, x) = HalfInteger(round(Int, 2x), 2)
Base.:*(a::HalfInteger, b::HalfInteger) = (a.numerator * b.numerator) // 4 # converts to Rational

include("combinatorics.jl")
include("linearalgebra.jl")
include("vectorspaces.jl")
include("compositeoperators.jl")
include("tensorproducts.jl")
include("fockspace.jl")
include("angularmomentum.jl")
include("lscoupling.jl")

end # module
