#
# Implements example 4.2 in Sarkka book, but due to technical reasons with
# regard to handling univariate and multivariate gaussians consistently,
# we simulate two random walks by making A, H, Q, R matrices as opposed to
# scalars specified in book.
#

function Example_4_2(; T = 400, seed = 1)

    # Set random number generator

    rg = MersenneTwister(seed)


    # Matrices in the dynamic model are:

    A = H = diagm(ones(2))
    Q = R = diagm(ones(2))

    # initialise

    m₀, P₀ = zeros(2), diagm(ones(2))

    x₀ = rand(rg, MvNormal(m₀, P₀))


    # store measurements and states

    yclean = Array{Vector{Float64},1}(undef, T)

    y = Array{Vector{Float64},1}(undef, T)

    x = Array{Vector{Float64},1}(undef, T)

    # first state

    x[1]      = A * x₀

    y[1]      = rand(rg, MvNormal(H * x[1], R))

    yclean[1] = H * x₀


    for k in 2:T

        x[k] = rand(rg, MvNormal(A * x[k-1], Q)) # (4.32)

        yclean[k] = H * x[k]

        y[k] = rand(rg, MvNormal(H * (x[k]), R)) # (4.33)

    end

    return yclean, y, x, A, H, Q, R, m₀, P₀

end


function runExample_4_2(seed = 1)

    # Simulate data

    yclean, y, x, A, H, Q, R, m₀, P₀ = Example_4_2(seed=seed)


    # Inference

    μfilter, Σfilter = filteringrecursion(y; A = A, H = H, Q = Q, R = R, m₀ = m₀, P₀ = P₀)

    μsmooth, Σsmooth = smoothingrecursion(y; A = A, H = H, Q = Q, R = R, m = μfilter, P = Σfilter)


    # collect positions

    truepositions = reduce(hcat, [xₖ[1:2] for xₖ in x])

    filtpositions = reduce(hcat, [xₖ[1:2] for xₖ in μfilter])

    smoopositions = reduce(hcat, [xₖ[1:2] for xₖ in μsmooth])


    # plot positions

    figure(1); cla()

    plot(truepositions[1,:], "-", label = "true")

    plot(filtpositions[1,:], "-", label = "filter")

    plot(smoopositions[1,:], "-", label = "smoother")

    legend()

    nothing

end
