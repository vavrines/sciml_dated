using Solaris, OrdinaryDiffEq
using Solaris.Optimization, Solaris.Flux
using SciMLSensitivity

datasize = 30
X = randn(4, datasize)
Y = randn(3, datasize)

ann = FnChain(FnDense(4,4,tanh), FnDense(4,3))
p = init_params(ann)

callback = function (p,l)
    display(l)
    return false
end

function loss_adjoint(fullp, time_batch, batch)
    pred = ann(time_batch, fullp)
    sum(abs2, batch .- pred)
end

loss_adjoint(p, X, Y)

k = 10
#train_loader = Flux.Data.DataLoader((ode_data, t), batchsize = k)
dl = Flux.Data.DataLoader((X, Y), batchsize = k)

numEpochs = 2

optfun = OptimizationFunction((θ, p, batch, time_batch) -> loss_adjoint(θ, batch, time_batch), Optimization.AutoReverseDiff())
optprob = OptimizationProblem(optfun, p)
using IterTools: ncycle
res1 = Optimization.solve(optprob, ADAM(0.05), ncycle(dl, numEpochs), callback = callback)
