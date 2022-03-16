#----------------------------#
# see page 57 in Sarkka book #
#----------------------------#

function filteringrecursion(y; A = A, H = H, Q = Q, R = R, m₀ = m₀, P₀ = P₀)

    N = length(y)

    # Store here state distribution

    μstate, Σstate = Array{Vector{Float64}}(undef, N), Array{Matrix{Float64}}(undef, N)

    # Start recursion with prior mean and prior covariance

    mₖ₋₁, Pₖ₋₁ = m₀, P₀

    for k in 1:N

        # prediction step

        mₖ⁻ = A * mₖ₋₁

        Pₖ⁻ = A * Pₖ₋₁ * A' + Q     # (4.20)

        # update step

        𝐯ₖ = y[k] - H * mₖ⁻

        Sₖ = H * Pₖ⁻ * H' + R

        Kₖ = (Pₖ⁻ * H') / Sₖ

        mₖ = mₖ⁻ + Kₖ * 𝐯ₖ

        Pₖ = Pₖ⁻ - Kₖ * Sₖ * (Kₖ)'  # (4.21)

        # store mean and covariance of Gaussian distribution N(xₖ|mₖ, Pₖ)

        Pₖ = (Pₖ+(Pₖ)') / 2

        μstate[k], Σstate[k] = mₖ, Pₖ

        # next state

        mₖ₋₁, Pₖ₋₁ = mₖ, Pₖ

    end

    return μstate, Σstate

end


function testfiltering() # Works ✅

    yclean, y, x, A, H, Q, R, m₀, P₀ = simulatedata()

    μ, Σ = filteringrecursion(y; A = A, H = H, Q = Q, R = R, m₀ = m₀, P₀ = P₀)

    figure()

    subplot(211)
    plot([yᵢ[1]    for yᵢ in yclean], label = "simulated 1 clean")
    plot([yᵢ[1]    for yᵢ in y], label = "simulated 1")
    plot([(H*m)[1] for m  in μ], label = "mean 1")
    legend()
    
    subplot(212)
    plot([yᵢ[2]    for yᵢ in y], label = "simulated 2")
    plot([(H*m)[2] for m  in μ], label = "mean 2")

    legend()

end
