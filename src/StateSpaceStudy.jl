module StateSpaceStudy

    using LinearAlgebra, Random, Distributions, PyPlot

    include("filteringrecursion.jl")

    include("smoothingrecursion.jl")

    include("simulatedata.jl")

    export simulatedata, filteringrecursion, smoothingrecursion

end
