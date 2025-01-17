"""
Architecture II

H-theorem preserving RelaxNet
"""

using Kinetic, Solaris
using KitBase.Distributions, KitBase.JLD2
using Solaris.Optimization, ReverseDiff
using Solaris.Flux: throttle, Adam, Data, elu, relu
using Solaris.Optim: LBFGS
using Base.Threads: @threads

cd(@__DIR__)
include("../../nn.jl")

#isNewRun = true
isNewRun = false

set = config_ntuple(
    u0 = -5,
    u1 = 5,
    nu = 80,
    v0 = -5,
    v1 = 5,
    nv = 28,
    w0 = -5,
    w1 = 5,
    nw = 28,
)

vs = VSpace1D(set.u0, set.u1, set.nu)
vs2 = VSpace2D(set.v0, set.v1, set.nv, set.w0, set.w1, set.nw)
vs3 = VSpace3D(set.u0, set.u1, set.nu, set.v0, set.v1, set.nv, set.w0, set.w1, set.nw)

if isNewRun
    m = moment_basis(vs.u, 4)

    pf = Normal(0.0, 0.005)
    pn = Uniform(0.1, 10)
    pt = Uniform(0.1, 5)

    pdfs = []
    for iter = 1:5000
        _f = sample_pdf(m, 4, [rand(pn), 0, 1/rand(pt)], pf)
        push!(pdfs, _f)
    end

    #=δ = heaviside.(vs.u)
    pdf1s = []
    for i = 1:length(pdfs)÷2
        _f = pdfs[i] .* δ + pdfs[length(pdfs) + 1 - i] .* (1 .- δ)
        push!(pdf1s, _f)
    end=#
    pdf1s = pdfs

    X0 = Array{Float64}(undef, vs3.nu, vs3.nv, vs3.nw, length(pdf1s))
    for iter in axes(X0)[end]
        w = moments_conserve(pdf1s[iter], vs.u, vs.weights)
        prim = conserve_prim(w, 3)
        for i = 1:vs.nu
            X0[i, :, :, iter] .= pdf1s[iter][i] * exp.(-vs3.v[i, :, :] .^ 2) .* exp.(-vs3.w[i, :, :] .^ 2)
        end
    end

    X = Array{Float64}(undef, vs.nu*2, length(pdf1s))
    for i in axes(X, 2)
        @assert moments_conserve(X0[:, :, :, i], vs3.u, vs3.v, vs3.w, vs3.weights)[1] < 50
        X[1:vs.nu, i], X[vs.nu+1:end, i] = reduce_distribution(X0[:, :, :, i], vs3.v, vs3.w, vs2.weights)
    end

    pk = Uniform(0.001, 1)
    kns = rand(pk, length(pdf1s))

    ϕ, ψ, χ = kernel_mode(5, vs3.u1, vs3.v1, vs3.w1, vs3.du[1], vs3.dv[1], vs3.dw[1],
        vs3.nu, vs3.nv, vs3.nw, set.α)

    Y0 = Array{Float64}(undef, vs3.nu, vs3.nv, vs3.nw, length(pdf1s))
    @threads for j in axes(Y0)[end]
        μ = ref_vhs_vis(kns[j], set.α, set.ω)
        kn_bz = hs_boltz_kn(μ, set.α)

        Q = @view Y0[:, :, :, j]
        boltzmann_fft!(Q, X0[:, :, :, j], kn_bz, 5, ϕ, ψ, χ)
    end

    Y = Array{Float64}(undef, vs.nu*2, length(pdf1s))
    for i in axes(Y, 2)
        Y[1:vs.nu, i], Y[vs.nu+1:end, i] = reduce_distribution(Y0[:, :, :, i], vs3.v, vs3.w, vs2.weights)
    end

    τ = Array{Float64}(undef, 1, size(X, 2))
    for i in axes(τ, 2)
        μ = ref_vhs_vis(kns[i], set.α, set.ω)
        w = moments_conserve(X[1:vs.nu, i], X[vs.nu+1:end, i], vs.u, vs.weights)
        prim = conserve_prim(w, 5/3)
        τ[1, i] = vhs_collision_time(prim, μ, set.ω)
    end
    X = vcat(X, τ)

    nm = vs.nu*2
    nν = vs.nu*2 + 1

    mn = FnChain(FnDense(nm, nm, tanh; bias = false), FnDense(nm, nm, tanh; bias = false), FnDense(nm, 6; bias = false))
    νn = FnChain(FnDense(nν, nν, tanh; bias = false), FnDense(nν, nν, tanh; bias = false), FnDense(nν, nm; bias = false))
    nn = BGKNet(mn, νn)
    u = init_params(nn)
else
    @load "prototype2.jld2" nn u X Y
end

L = size(Y, 2)

function loss(p, x, y)
    pred = nn(x, p, vs, 5/3, VDF{2,1}, Class{2})
    return sum(abs2, pred - y) / L
end
loss(p) = loss(p, X, Y)

lold = loss(u)

cb = function (p, l)
    println("loss: $(l)")
    return false
end
cb = throttle(cb, 1)

dl = Data.DataLoader((X, Y), batchsize = 1000, shuffle = true)
res = sci_train(loss, u, dl, Adam(); cb = cb, ad = Optimization.AutoReverseDiff(), iters = 500, epochs = 10)
res = sci_train(loss, res.u, dl, Adam(); cb = cb, ad = Optimization.AutoReverseDiff(), iters = 500, epochs = 20)

res = sci_train(loss, res.u, Adam(1e-4); cb = cb, ad = Optimization.AutoReverseDiff(), iters = 100)

u = res.u
@save "prototype2.jld2" nn u X Y
