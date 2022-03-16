function Example_4_3(; T = 200, seed = 1)

    # Set random number generator

    rg = MersenneTwister(seed)


    # state x is four dimensional, comprises car position and velocities

    Δt  = 1 / 10
    σ₁  = σ₂  = 1 / 2
    qᶜ₁ = qᶜ₂ = 1


    # Matrices in the dynamic model are:

    A = [1  0  Δt  0;
         0  1   0  Δt;
         0  0   1  0;
         0  0   0  1]

    Q = [(qᶜ₁*Δt^3)/3         0        (qᶜ₁*Δt^2)/2       0        ;
              0         (qᶜ₂*Δt^3)/3         0        (qᶜ₁*Δt^2)/2 ;
         (qᶜ₁*Δt^2)/2         0            qᶜ₁*Δt         0        ;
              0         (qᶜ₂*Δt^2)/2         0          qᶜ₂*Δt     ]

    # Measurement matrices

    H = [1 0 0 0;
         0 1 0 0]

    R = [σ₁^2  0;
         0    σ₂^2]

    # initialise

    m₀, P₀ = zeros(4), 0.1*Matrix(I, 4, 4)

    x₀ = rand(MvNormal(m₀, P₀))


    # store measurements and states

    yclean = Array{Vector{Float64},1}(undef, T)

    y = Array{Vector{Float64},1}(undef, T)

    x = Array{Vector{Float64},1}(undef, T)

    # first state

    x[1] = A * x₀

    y[1] = rand(rg, MvNormal(H * x[1], R))
    

    for k in 2:T

        x[k] = rand(rg, MvNormal(A * x[k-1], Q)) # (4.32)

        yclean[k] = H * x[k]

        y[k] = rand(rg, MvNormal(H * x[k], R)) # (4.33)

    end

    return yclean, y, x, A, H, Q, R, m₀, P₀

end


function runExample_4_3(seed = 1)

    # Simulate data

    yclean, y, x, A, H, Q, R, m₀, P₀ = Example_4_3(seed=seed)


    # Inference

    μfilter, Σfilter = filteringrecursion(y; A = A, H = H, Q = Q, R = R, m₀ = m₀, P₀ = P₀)

    μsmooth, Σsmooth = smoothingrecursion(y; A = A, H = H, Q = Q, R = R, m = μfilter, P = Σfilter)


    # collect positions

    truepositions = reduce(hcat, [xₖ[1:2] for xₖ in x])

    filtpositions = reduce(hcat, [xₖ[1:2] for xₖ in μfilter])

    smoopositions = reduce(hcat, [xₖ[1:2] for xₖ in μsmooth])


    # plot positions

    figure(1)

    subplot(211); cla()

    plot(truepositions[1,:], truepositions[2,:], "o", label = "true positions")

    plot(filtpositions[1,:], filtpositions[2,:], ".", label = "filter positions", alpha=0.5)

    legend()

    subplot(212); cla()

    plot(truepositions[1,:], truepositions[2,:], "o", label = "true positions")

    plot(smoopositions[1,:], smoopositions[2,:], ".", label = "smooth positions", alpha=0.5)

    legend()

    nothing

end
