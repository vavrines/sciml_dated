using DiffEqFlux, OrdinaryDiffEq, Flux, Optim, Plots

tspan = (0.0f0, 1.5f0)
tlen = 30
t = range(tspan[1], tspan[2], length = tlen)

n = 2  # number of ODEs
d = 2  # number of data pairs

X = Array{Float32}(undef, n, d)
X[:, 1] .= [2.0, 0.0]
X[:, 2] .= [1.5, 0.5]
#X[:, 3] .= [1., 0.8]

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
    FastDense(2, 50, tanh),
    #FastDense(50, 50, tanh),
    FastDense(50, 2)
)
node = NeuralODE(dudt, tspan, Tsit5(), saveat = t)

function loss_node(p, x, y)
    pred = node(x, p)
    loss = sum(abs2, y .- pred)
    loss
end

function loss_node(p)
    loss = 0.
    for d in data
        loss += sum(abs2, d[2] .- node(d[1], p))
    end
    return loss
end

cb = function (p, l) #callback function to observe training
    display(l)
    return false
end

res = DiffEqFlux.sciml_train(
    loss_node,
    node.p,
    ADAM(0.05),
    #dataset,
    cb = Flux.throttle(cb, 1),
    maxiters = 1000,
)

res = DiffEqFlux.sciml_train(
    loss_node,
    res.minimizer,
    ADAM(),
    #dataset,
    cb = Flux.throttle(cb, 1),
    maxiters = 1000,
)

res = DiffEqFlux.sciml_train(
    loss_node,
    res.minimizer,
    LBFGS(),
    #dataset,
    cb = cb,
    maxiters = 500,
)

plot(node(X[:, 1], res.minimizer))
plot!(t, Y[1, :, 1])
plot!(t, Y[2, :, 1])
