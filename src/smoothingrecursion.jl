#-----------------------------#
# see page 136 in Sarkka book #
#-----------------------------#

function smoothingrecursion(y; A = A, H = H, Q = Q, R = R, m = m, P = P)

    N = length(y); @assert(N == length(m) == length(P))

    # Store here state distribution

    μstate, Σstate = Array{Vector{Float64}}(undef, N), Array{Matrix{Float64}}(undef, N)

    # Start recursion with prior mean and prior covariance

    mˢₖ₊₁, Pˢₖ₊₁ = m[end], P[end]

    μstate[N], Σstate[N] = mˢₖ₊₁, Pˢₖ₊₁

    for k in N-1:-1:1

        m⁻ₖ₊₁ = A * m[k]

        P⁻ₖ₊₁ = A * P[k] * A' + Q

        Gₖ = (P[k] * A') / P⁻ₖ₊₁

        mˢₖ = m[k] + Gₖ * (mˢₖ₊₁ - m⁻ₖ₊₁)

        Pˢₖ = P[k] + Gₖ * (Pˢₖ₊₁ - P⁻ₖ₊₁) * (Gₖ)'

        # store mean and covariance of Gaussian distribution N(xₖ|mˢₖ, Pˢₖ)
        Pˢₖ = (Pˢₖ + (Pˢₖ)') / 2

        μstate[k], Σstate[k] = mˢₖ, Pˢₖ

        # previous state

        mˢₖ₊₁, Pˢₖ₊₁ = mˢₖ, Pˢₖ

    end

    return μstate, Σstate

end



function testsmoothing(seed=1)

    yclean, y, x, A, H, Q, R, m₀, P₀ = simulatedata(seed=seed)

    μfilter, Σfilter = filteringrecursion(y; A = A, H = H, Q = Q, R = R, m₀ = m₀, P₀ = P₀)

    μsmooth, Σsmooth = smoothingrecursion(y; A = A, H = H, Q = Q, R = R, m = μfilter, P = Σfilter)


    figure()
    subplot(211)
    plot([yᵢ[1]        for yᵢ in yclean], label = "simulated 1")
    plot([(H*xᵢ)[1]    for xᵢ in x], label = "simulated 1")
    legend()

    subplot(212)
    plot([xᵢ[1]    for xᵢ in x], label = "simulated 1")
    plot([xᵢ[1]    for xᵢ in μfilter], label = "filtered 1")
    plot([xᵢ[1]    for xᵢ in μsmooth], "--", label = "smoothed 1")
    legend()

    acc = accfilter = accsmooth = 0.0

    for i in 1:length(y)

        acc += sum((yclean[i] - H*x[i]).^2) / length(y)

        accfilter += sum((yclean[i] - H*μfilter[i]).^2)/ length(y)

        accsmooth += sum((yclean[i] - H*μsmooth[i]).^2)/ length(y)

    end

    @show acc, accfilter, accsmooth

    accx = accfilterx = accsmoothx = 0.0

    for i in 1:length(y)

        accx += sum((x[i] - x[i]).^2)

        accfilterx += sum((x[i] - μfilter[i]).^2)

        accsmoothx += sum((x[i] - μsmooth[i]).^2)

    end

    @show accx, accfilterx, accsmoothx


    x, μfilter, μsmooth

end
