using Kinetic, Solaris, OrdinaryDiffEq, Plots
using Kinetic.KitBase.Distributions, Kinetic.KitBase.JLD2
using Solaris.Optimization, ReverseDiff
using Solaris.Flux: sigmoid, Adam, relu, cpu, gpu
using Solaris.Optim: LBFGS
using KitBase.ProgressMeter: @showprogress
using Base.Threads: @threads
using KitBase: AV, AM
using KitBase.CUDA

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
m = moment_basis(vs.u, 5)
momentquad = vs.u .* vs.weights
energyquad = vs.u .^ 2 .* vs.weights

pf = Normal(0.0, 0.01)
pn = Uniform(0.1, 10)
pt = Uniform(0.1, 8)

pdfs = []
for iter = 1:10000
    _f = sample_pdf(m, [rand(pn), 0, 1/rand(pt)], pf)
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

Y = Array{Float32}(undef, vs.nu, size(X, 2))
for j in axes(Y, 2)
    w = moments_conserve(pdfs[j], vs.u, vs.weights)
    prim = conserve_prim(w, 3)
    M = maxwellian(vs.u, prim)
    q = heat_flux(pdfs[j], prim, vs.u, vs.weights)
    S = shakhov(vs.u, M, q, prim, 2/3) # Pr shouldn't be 0
    @. Y[:, j] = (M + S - pdfs[j]) / τ[1, j]
    #@. Y[:, j] = (M - pdfs[j]) / τ[1, j]
end

function shuffle_data(X, Y, n = 1000)
    tmp = rand() * 8000 |> round |> Int
    idx = tmp:tmp + (n - 1)
    data = (X[:, idx], Y[:, idx])

    return data
end

nm = vs.nu
nν = vs.nu + 1

mn = FnChain(FnDense(nm, nm*2, tanh; bias = false), FnDense(nm*2, nm*4, tanh; bias = false), FnDense(nm*4, nm; bias = false))
νn = FnChain(FnDense(nν, nν*2, tanh; bias = false), FnDense(nν*2, nν*4, tanh; bias = false), FnDense(nν*4, nm; bias = false))
#mn = FnChain(FnDense(vs.nu, vs.nu * 2, tanh; bias = false), FnDense(vs.nu * 2, vs.nu; bias = false))
#νn = FnChain(FnDense(vs.nu + 1, vs.nu * 2 + 1, tanh; bias = false), FnDense(vs.nu * 2 + 1, vs.nu; bias = false))
nn = BGKNet(mn, νn)
p = begin
    if isNewRun
        init_params(nn)
    else
        @load "start.jld2" u
        u
    end
end

data = shuffle_data(X, Y)

function loss(p)
    L = size(data[1], 2)
    pred = nn(data[1], p, vs)
    r1 = sum(abs2, discrete_moments(pred, vs.weights)) / L
    r2 = sum(abs2, discrete_moments(pred, momentquad)) / L
    r3 = sum(abs2, discrete_moments(pred, energyquad)) / L

    return sum(abs2, pred - data[2]) / L + (r1 + r2 + r3) * 1e-6
    #return (r1 + r2 + r3) * 1e-6
end

loss(p)

#his = []
cb = function (p, l)
    println("loss: $(loss(p))")
    #push!(his, l)
    return false
end

res = sci_train(loss, p, Adam(); cb = cb, ad = Optimization.AutoReverseDiff(), iters = 200)
lmin = loss(res.u)

for epoch = 1:100
    #data = shuffle_data(X, Y)
    res = sci_train(loss, res.u, Adam(1e-4); cb = cb, ad = Optimization.AutoReverseDiff(), iters = 500)

    if loss(res.u) < lmin
        u = res.u
        @save "minimizer.jld2" u
    end
end

#res = sci_train(loss, res.u, LBFGS(); cb = cb, ad = Optimization.AutoReverseDiff(), iters = 100)

u = res.u
@save "minimizer.jld2" u