using DiffEqFlux, OrdinaryDiffEq, Flux, Optim, Plots

tspan = (0.0f0, 1.5f0)
tlen = 30
t = range(tspan[1], tspan[2], length = tlen)

n = 2  # number of ODEs
d = 3  # number of data pairs

X = Array{Float32}(undef, n, d)
X[:, 1] .= [2.0, 0.0]
X[:, 2] .= [1.5, 0.5]
X[:, 3] .= [2., 0.]

function trueODEfunc(du, u, p, t)
    true_A = [-0.1 2.0; -2.0 -0.1]
    du .= ((u .^ 3)'true_A)'
end

Y = Array{Float32}(undef, n, tlen, d)
for k = 1:d
    prob = ODEProblem(trueODEfunc, X[:, k], tspan)
    ode_data = Array(solve(prob, Tsit5(), saveat = t))
    Y[:, :, k] .= ode_data
end

#pic = plot(t, Y[1, :, 1]); plot!(pic, t, Y[2, :, 1]); display(pic)
#pic = plot(t, Y[1, :, 2]); plot!(pic, t, Y[2, :, 2]); display(pic)
#pic = plot(t, Y[1, :, 3]); plot!(pic, t, Y[2, :, 3]); display(pic)

data = [(X[:, i], Y[:, :, i]) for i = 1:d]
dataset = Iterators.cycle(data)

dudt = FastChain(
    (x, p) -> x .^ 3,
    FastDense(2, 20, tanh),
    FastDense(20, 20, tanh),
    FastDense(20, 2)
)
node = NeuralODE(dudt, tspan, Tsit5(), saveat = t)

function loss_node(p, x, y)
    pred = node(x, p)
    loss = sum(abs2, y .- pred)
    loss, pred
end

cb = function (p, l, pred; doplot = false) #callback function to observe training
    display(l)
    # plot current prediction against data
    if doplot
        pl = scatter(t, ode_data[1, :], label = "data")
        scatter!(pl, t, pred[1, :], label = "prediction")
        display(plot(pl))
    end
    return false
end

res = DiffEqFlux.sciml_train(
    loss_node,
    node.p,
    ADAM(0.05),
    dataset,
    cb = Flux.throttle(cb, 1),
    maxiters = 1000,
)

res = DiffEqFlux.sciml_train(
    loss_node,
    res.minimizer,
    ADAM(),
    dataset,
    cb = Flux.throttle(cb, 1),
    maxiters = 2000,
)

res = DiffEqFlux.sciml_train(
    loss_node,
    res.minimizer,
    LBFGS(),
    dataset,
    cb = cb,
    maxiters = 500,
)

plot(node(X[:, 1], res.minimizer))
plot!(t, Y[1, :, 1])
plot!(t, Y[2, :, 1])
