using Solaris, OrdinaryDiffEq, SciMLSensitivity
using Solaris.Optimization
using Solaris.Flux: Data, Adam
using IterTools: ncycle

datasize = 30
X = randn(4, datasize)
Y = randn(3, datasize)

ann = FnChain(FnDense(4, 4, tanh), FnDense(4, 3))
p = init_params(ann)

cb = function (p, l)
    display(l)
    return false
end

function loss(p, x, y)
    pred = ann(x, p)
    sum(abs2, y .- pred)
end

loss(p, X, Y)

dl = Data.DataLoader((X, Y), batchsize = 10)

numEpochs = 2

optfun = OptimizationFunction((θ, p, x, y) -> loss(θ, x, y), Optimization.AutoReverseDiff())
optprob = OptimizationProblem(optfun, p)
res = Optimization.solve(optprob, Adam(0.05), ncycle(dl, numEpochs), callback = cb)

# this is equivalent as Solaris.sci_train
res = sci_train(loss, res.u, dl, Adam(0.05); callback = cb)
