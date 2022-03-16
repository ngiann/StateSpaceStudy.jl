#----------------------------#
# see page 57 in Sarkka book #
#----------------------------#

function simulatedata(N = 1_000; seed = 1)

    rg = MersenneTwister(seed)

    A, H = 0.5*randn(rg, 2, 2),   0.5*randn(rg, 2, 2)
    Q, R = 0.005*Matrix(I, 2, 2), 0.05*Matrix(I, 2, 2)
    m₀   = 0.5*randn(rg, 2)
    P₀   = 0.05*Matrix(I, 2, 2)

    yclean = Array{Vector{Float64}, 1}(undef, N)
    y = Array{Vector{Float64}, 1}(undef, N)
    x = Array{Vector{Float64}, 1}(undef, N)

    x[1] = rand(rg, MvNormal(m₀, P₀))
    yclean[1] = H*x[1]
    y[1] = rand(rg, MvNormal(H * x[1], R))

    for k in 2:N

        # prediction step

        x[k] = rand(rg, MvNormal(A * x[k-1], Q))

        yclean[k] = H * x[k]

        y[k] = rand(rg, MvNormal(H * x[k], R))

    end

    return yclean, y, x, A, H, Q, R, m₀, P₀

end
