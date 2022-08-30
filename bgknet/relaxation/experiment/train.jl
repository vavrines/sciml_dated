using Kinetic, Solaris
using KitBase.Distributions, KitBase.JLD2
using Solaris.Optimization, ReverseDiff
using Solaris.Flux: throttle, Adam, Data
using Solaris.Optim: LBFGS
using Base.Threads: @threads

cd(@__DIR__)

#isNewRun = true
isNewRun = false

set = (
    u0 = -5,
    u1 = 5,
    nu = 80,
    v0 = -5,
    v1 = 5,
    nv = 28,
    w0 = -5,
    w1 = 5,
    nw = 28,
    nm = 5,
    K = 0,
    alpha = 1.0,
    omega = 0.5,
)

vs = VSpace1D(set.u0, set.u1, set.nu)
vs2 = VSpace2D(set.v0, set.v1, set.nv, set.w0, set.w1, set.nw)
vs3 = VSpace3D(set.u0, set.u1, set.nu, set.v0, set.v1, set.nv, set.w0, set.w1, set.nw)

if isNewRun
    m = moment_basis(vs.u, 4)

    pf = Normal(0.0, 0.005)
    pn = Uniform(0.01, 2)
    pt = Uniform(0.1, 2)

    pdfs = []
    for iter = 1:5000
        _f = sample_pdf(m, 4, [rand(pn), 0, 1/rand(pt)], pf)
        push!(pdfs, _f)
    end

    X0 = Array{Float64}(undef, vs3.nu, vs3.nv, vs3.nw, length(pdfs))
    for iter in axes(X0)[end]
        w = moments_conserve(pdfs[iter], vs.u, vs.weights)
        prim = conserve_prim(w, 3)
        for i = 1:vs.nu
            X0[i, :, :, iter] .= pdfs[iter][i] * exp.(-prim[end] .* vs3.v[i, :, :] .^ 2) .* exp.(-prim[end] .* vs3.w[i, :, :] .^ 2)
        end
    end

    X = Array{Float64}(undef, vs.nu*2, length(pdfs))
    for i in axes(X, 2)
        @assert moments_conserve(X0[:, :, :, i], vs3.u, vs3.v, vs3.w, vs3.weights)[1] < 50
        X[1:vs.nu, i], X[vs.nu+1:end, i] = reduce_distribution(X0[:, :, :, i], vs3.v, vs3.w, vs2.weights)
    end

    pk = Uniform(0.001, 1)
    kns = rand(pk, length(pdfs))

    ϕ, ψ, χ = kernel_mode(5, vs3.u1, vs3.v1, vs3.w1, vs3.du[1], vs3.dv[1], vs3.dw[1],
        vs3.nu, vs3.nv, vs3.nw, set.alpha)

    Y0 = Array{Float64}(undef, vs3.nu, vs3.nv, vs3.nw, length(pdfs))
    @threads for j in axes(Y0)[end]
        #println("$j of $(length(pdfs))")
        μ = ref_vhs_vis(kns[j], set.alpha, set.omega)
        kn_bz = hs_boltz_kn(μ, set.alpha)

        Q = @view Y0[:, :, :, j]
        boltzmann_fft!(Q, X0[:, :, :, j], kn_bz, 5, ϕ, ψ, χ)
    end

    Y = Array{Float64}(undef, vs.nu*2, length(pdfs))
    for i in axes(Y, 2)
        Y[1:vs.nu, i], Y[vs.nu+1:end, i] = reduce_distribution(Y0[:, :, :, i], vs3.v, vs3.w, vs2.weights)
    end

    τ = Array{Float64}(undef, 1, size(X, 2))
    for i in axes(τ, 2)
        μ = ref_vhs_vis(kns[i], set.alpha, set.omega)
        w = moments_conserve(X[1:vs.nu, i], X[vs.nu+1:end, i], vs.u, vs.weights)
        prim = conserve_prim(w, 5/3)
        τ[1, i] = vhs_collision_time(prim, μ, set.omega)
    end
    X = vcat(X, τ)

    nm = vs.nu*2
    nν = vs.nu*2 + 1

    mn = FnChain(FnDense(nm, nm, tanh; bias = false), FnDense(nm, nm, tanh; bias = false), FnDense(nm, nm; bias = false))
    νn = FnChain(FnDense(nν, nν, tanh; bias = false), FnDense(nν, nν, tanh; bias = false), FnDense(nν, nm; bias = false))
    nn = BGKNet(mn, νn)
    u = init_params(nn)
else
    @load "prototype.jld2" nn u X Y
end

L = size(Y, 2)

function loss(p, x, y)
    pred = nn(x, p, vs, 5/3)
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

res = sci_train(loss, u, dl, Adam(); cb = cb, ad = Optimization.AutoReverseDiff(), iters = 500, epochs = 50)
res = sci_train(loss, res.u, dl, Adam(); cb = cb, ad = Optimization.AutoReverseDiff(), iters = 500, epochs = 100)
res = sci_train(loss, res.u, dl, LBFGS(); cb = cb, ad = Optimization.AutoReverseDiff(), iters = 500, epochs = 10)

lnew = loss(res.u)

if lnew < lold
    u = res.u
    @save "prototype.jld2" nn u X Y
    run(`cp prototype.jld2 prototype_backup.jld2`)
end

#idx = rand() * length(pdfs) |> round |> Int
#contour(vs3.u[:, 1, 1], vs3.v[1, :, 1], vs3.w[1, 1, :], X0[:, :, :, idx], alpha = 0.3)

res = sci_train(loss, res.u, Adam(); cb = cb, ad = Optimization.AutoReverseDiff(), iters = 200)

using Plots

begin
    idx = rand() * 5000 |> round |> Int
    pred = nn(X[:, idx], res.u, vs, 5/3)
    plot(X[1:end-1, idx])
    plot!(Y[:, idx])
    plot!(pred)
end
