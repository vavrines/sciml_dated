using KitML, CUDA, OrdinaryDiffEq
using KitML.Flux, KitML.DiffEqFlux

nn = FastChain(FastDense(2, 4, tanh), FastDense(4, 2))
p = initial_params(nn) |> gpu

X = CUDA.randn(2, 10)
Y = CUDA.rand(2, 10, 2)

function dfdt(f, p, t)
    df = exp.(f) + nn(f, p)
    #df = nn(f, p)
end
prob = ODEProblem(dfdt, X, (0.0, 1.0), p)

function loss(p)
    #sol = nn(X, p)
    sol = solve(prob, Euler(), u0=X, p=p, dt=1.0) |> gpu
    #sol = dfdt(X, p, 0.0) .* dt .+ X
    return sum(abs2, sol - Y)
end

cb = function (p, l)
    println("loss: $l")
    return false
end

loss(p)

res = sci_train(loss, p, ADAM(); cb=Flux.throttle(cb, 1), maxiters=200)
