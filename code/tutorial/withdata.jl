using DiffEqFlux, OrdinaryDiffEq, Flux, Optim, Plots

n = 2  # number of ODEs
tspan = (0.0, 1.0)

d = 3  # number of data pairs
X = rand(n, d)
Y = rand(n, d)
data = [(X[:, i], Y[:, i]) for i = 1:d]
dataset = Iterators.cycle(data)

NN = FastChain(FastDense(n, 10n, tanh), FastDense(10n, n))
node = NeuralODE(NN, tspan, Tsit5(), reltol = 1e-4, saveat = [tspan[end]])

function loss(θ, x, y)
    pred = node(x, θ)
    data = y
    loss = sum(abs2, y .- x)
    return loss
end

cb = function (p, l)
    print("loss: ")
    display(l)
    return false
end

res = DiffEqFlux.sciml_train(
    loss,
    node.p,
    ADAM(),
    dataset,
    cb = cb,
    maxiters = 10,
)

res = DiffEqFlux.sciml_train(
    loss,
    res.minimizer,
    ADAM(),
    dataset,
    cb = cb,
    maxiters = 10,
)
