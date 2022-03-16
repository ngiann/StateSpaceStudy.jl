module StateSpaceStudy

    using LinearAlgebra, Random, Distributions, PyPlot

    include("filteringrecursion.jl")

    include("smoothingrecursion.jl")

    include("simulatedata.jl")

    include("Example_4_3.jl")

    include("Example_4_2.jl")

    export simulatedata, filteringrecursion, smoothingrecursion

end
