using Kinetic, Solaris
using Kinetic.KitBase.Distributions, Kinetic.KitBase.JLD2
using Solaris.Optimization, ReverseDiff
using Solaris.Flux: sigmoid, Adam, relu, cpu, gpu, Data, throttle
using Solaris.Optim: LBFGS
using IterTools: ncycle

cd(@__DIR__)

#isNewRun = true
isNewRun = false

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
    m = moment_basis(vs.u, 4)

    pf = Normal(0.0, 0.01)
    pn = Uniform(0.1, 10)
    pt = Uniform(0.1, 8)

    pdfs = []
    for iter = 1:10000
        _f = sample_pdf(m, 4, [rand(pn), 0, 1/rand(pt)], pf)
        push!(pdfs, _f)
    end

    pk = Uniform(0.001, 1)
    kns = rand(pk, length(pdfs))

    X = Array{Float64}(undef, vs.nu, length(pdfs))
    for i in axes(X, 2)
        @assert moments_conserve(pdfs[i], vs.u, vs.weights)[1] < 50
        X[:, i] .= pdfs[i]
    end

    τ = Array{Float64}(undef, 1, size(X, 2))
    for i in axes(τ, 2)
        μ = ref_vhs_vis(kns[i], set.alpha, set.omega)
        w = moments_conserve(pdfs[i], vs.u, vs.weights)
        prim = conserve_prim(w, 3)
        τ[1, i] = vhs_collision_time(prim, μ, set.omega)
    end
    X = vcat(X, τ)

    Y = Array{Float64}(undef, vs.nu, size(X, 2))
    for j in axes(Y, 2)
        w = moments_conserve(pdfs[j], vs.u, vs.weights)
        prim = conserve_prim(w, 3)
        M = maxwellian(vs.u, prim)
        q = heat_flux(pdfs[j], prim, vs.u, vs.weights)
        S = shakhov(vs.u, M, q, prim, 2/3) # Pr shouldn't be 0
        @. Y[:, j] = (M + S - pdfs[j]) / τ[1, j]
        #@. Y[:, j] = (M - pdfs[j]) / τ[1, j]
    end

    nm = vs.nu
    nν = vs.nu + 1

    mn = FnChain(FnDense(nm, nm, tanh; bias = false), FnDense(nm, nm, tanh; bias = false), FnDense(nm, nm; bias = false))
    νn = FnChain(FnDense(nν, nν, tanh; bias = false), FnDense(nν, nν, tanh; bias = false), FnDense(nν, nm; bias = false))
    nn = BGKNet(mn, νn)
    u = init_params(nn)
else
    @load "model.jld2" nn u X Y
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
    @save "model.jld2" nn u X Y
    run(`cp model.jld2 model_backup.jld2`)
end
