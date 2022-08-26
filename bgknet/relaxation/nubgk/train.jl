using Kinetic, Solaris
using Kinetic.KitBase.Distributions, Kinetic.KitBase.JLD2
using Solaris.Optimization, ReverseDiff
using Solaris.Flux: sigmoid, Adam, relu, cpu, gpu, Data, throttle
using Solaris.Optim: LBFGS

cd(@__DIR__)

isNewRun = true
#isNewRun = false

set = (
    u0 = -8,
    u1 = 8,
    nu = 80,
    K = 0,
    alpha = 1.0,
    omega = 0.5,
)

vs = VSpace1D(set.u0, set.u1, set.nu)
momentquad = vs.u .* vs.weights
energyquad = vs.u .^ 2 .* vs.weights

if isNewRun
    @load "../shakhov/prototype.jld2" nn u X

    τ1 = Array{Float64}(undef, vs.nu, size(X, 2))
    for i in axes(X, 2)
        w = moments_conserve(X[begin:end-1, i], vs.u, vs.weights)
        prim = conserve_prim(w, 3)
        _τ = ones(vs.nu) .* X[end, i]
        τ1[:, i] .= KB.νshakhov_relaxation_time(_τ, vs.u, prim)
    end

    Y = Array{Float64}(undef, vs.nu, size(X, 2))
    for j in axes(Y, 2)
        w = moments_conserve(X[begin:end-1, j], vs.u, vs.weights)
        prim = conserve_prim(w, 3)
        M = maxwellian(vs.u, prim)
        @. Y[:, j] = (M - X[begin:end-1, j]) / τ1[:, j]
    end
else
    @load "prototype.jld2" nn u X Y
end

L = size(Y, 2)

function loss(p, x, y)
    pred = nn(x, p, vs)
    #r1 = sum(abs2, discrete_moments(pred, vs.weights)) / L
    #r2 = sum(abs2, discrete_moments(pred, momentquad)) / L
    #r3 = sum(abs2, discrete_moments(pred, energyquad)) / L

    #return sum(abs2, pred - y) / L + (r1 + r2 + r3) * 1e-6
    return sum(abs2, pred - y) / L
end

lold = loss(u, X, Y)

#his = []
cb = function (p, l)
    println("loss: $(l)")
    #push!(his, l)
    return false
end
cb = throttle(cb, 1)

dl = Data.DataLoader((X, Y), batchsize = 2000, shuffle = true)

res = sci_train(loss, u, dl, Adam(); cb = cb, ad = Optimization.AutoReverseDiff(), iters = 1000)

res = sci_train(loss, res.u, dl, Adam(); cb = cb, ad = Optimization.AutoReverseDiff(), iters = 1000, epochs = 100)
res = sci_train(loss, res.u, dl, LBFGS(); cb = cb, ad = Optimization.AutoReverseDiff(), iters = 500, epochs = 20)

# equivalent low-level functions
#optfun = OptimizationFunction((θ, p, x, y) -> loss(θ, x, y), Optimization.AutoReverseDiff())
#optprob = OptimizationProblem(optfun, res.u)
#res1 = Optimization.solve(optprob, Adam(0.001), ncycle(dl, 2), callback = cb, maxiters = 20)

loss(p) = loss(p, X, Y)
lnew = loss(res.u)

#res = sci_train(loss, p, Adam(); cb = cb, ad = Optimization.AutoReverseDiff(), iters = 200)
#res = sci_train(loss, res.u, Adam(); cb = cb, ad = Optimization.AutoReverseDiff(), iters = 200)
#res = sci_train(loss, res.u, LBFGS(); cb = cb, ad = Optimization.AutoReverseDiff(), iters = 200)

if lnew < lold
    u = res.u
    @save "prototype.jld2" nn u X Y
    run(`cp prototype.jld2 prototype_backup.jld2`)
end
