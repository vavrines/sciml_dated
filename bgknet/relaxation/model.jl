using Kinetic, Solaris
using Solaris.Optimization, ReverseDiff
using Solaris.Flux: sigmoid, Adam
using Solaris.Optim: LBFGS

isNewStart = false

cd(@__DIR__)
@load "dataset.jld2" set X Y

vs = VSpace3D(set.u0, set.u1, set.nu, set.v0, set.v1, set.nv, set.w0, set.w1, set.nw; precision = Float32)
vs1d = VSpace1D(set.u0, set.u1, set.nu; precision = Float32)

mn = FnChain(FnDense(vs.nu, vs.nu * 2, tanh; bias = false), FnDense(vs.nu * 2, vs.nu; bias = false))
νn = FnChain(FnDense(vs.nu + 1, vs.nu * 2 + 1, tanh; bias = false), FnDense(vs.nu * 2 + 1, vs.nu, sigmoid; bias = false))
nn = BGKNet(mn, νn)

if isNewStart
    u = init_params(nn)
else
    @load "minimizer.jld2" u
end

data = (X, Y)
L = size(data[1], 2)
loss(p) = sum(abs2, nn(data[1], p, vs1d) - data[2]) / L

#his = []
cb = function (p, l)
    println("loss: $(loss(p))")
    #push!(his, l)
    return false
end

res = sci_train(loss, u, Adam(), Optimization.AutoReverseDiff(); cb = cb, maxiters = 5000)
res = sci_train(loss, res.u, LBFGS(), Optimization.AutoReverseDiff(); cb = cb, maxiters = 1000)

u = res.u
@save "minimizer.jld2" u
